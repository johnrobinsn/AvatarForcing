import os, copy, pickle, torch, yaml, random, json, cv2, torchvision, subprocess, argparse, uuid, datetime, tempfile, librosa, face_alignment, time

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

import albumentations as A
import albumentations.pytorch.transforms as A_pytorch

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from transformers import Wav2Vec2FeatureExtractor

class DataProcessor:
    def __init__(self, opt):
        self.opt = opt
        self.fps = opt.fps
        self.input_size = opt.input_size
        self.sampling_rate = opt.sampling_rate
        self.only_last_features = opt.only_last_features
        self.num_frames_for_clip = int(opt.sec * self.fps)

        # wav2vec2 audio preprocessor
        self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path, local_files_only=True)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        # image transform 
        self.transform = A.Compose([
                A.Resize(height = opt.input_size, width = opt.input_size, interpolation = cv2.INTER_AREA),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                A_pytorch.ToTensorV2()])

    def default_aud_loader(self, path: str) -> torch.Tensor:
        speech_array, sampling_rate = librosa.load(path, sr=self.sampling_rate)
        return self.wav2vec_preprocessor(speech_array, sampling_rate=sampling_rate, return_tensors='pt').input_values[0]

    def default_img_loader(self, path:str) -> np.ndarray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def preprocess_face(self, image_path: str, pad_ratio: float=1.0):
        image = self.default_img_loader(image_path)
        h, w = image.shape[0:2]
        mult = 360. / image.shape[0]

        resized_image = cv2.resize(image, dsize=(0, 0), fx = mult, fy = mult, interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)        
        bboxes = self.fa.face_detector.detect_from_image(resized_image)
        bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score) for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
        bboxes = bboxes[0]

        bsy = int((bboxes[3] - bboxes[1]) / 2)
        bsx = int((bboxes[2] - bboxes[0]) / 2)
        my  = int((bboxes[1] + bboxes[3]) / 2)
        mx  = int((bboxes[0] + bboxes[2]) / 2)

        bs = int(max(bsy, bsx) * (1+pad_ratio))
        x1, y1 = mx - bs, my - bs
        x2, y2 = mx + bs, my + bs
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        bsx, bsy = x2 - x1, y2 - y1
        mx, my = int(x1 + bsx // 2), int(y1 + bsy // 2)
        bs = int(min(bsx, bsy) // 2)

        face = image[my - bs: my + bs, mx-bs:mx + bs]
        face = cv2.resize(face, dsize=(self.opt.input_size, self.opt.input_size), interpolation = cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
        return face


    def preprocess(self, avatar_ref_path: str, avatar_audio_path: str, user_audio_path: str, user_video_path: str) -> dict:
        max_len    = int(30 * self.fps)            # maximum 30 seconds
        max_len_sr = int(30 * self.sampling_rate)  # maximum 30 seconds

        if os.path.exists(user_audio_path):
            user_a = self.default_aud_loader(user_audio_path)[:max_len_sr].unsqueeze(0)
        else:
            user_a = None

        avatar_a = self.default_aud_loader(avatar_audio_path)[:max_len_sr].unsqueeze(0)

        if os.path.exists(user_video_path):
            user_frames = []
            user_frame_paths = sorted(Path(user_video_path).rglob("*.jpg"))
            for frame_path in user_frame_paths:
                user_frame = self.default_img_loader(str(frame_path))
                user_frame = self.transform(image=user_frame)["image"].unsqueeze(0)
                user_frames.append(user_frame)

        else:
            user_frames = None

        # avatar_ref = self.default_img_loader(avatar_ref_path)
        avatar_ref = self.preprocess_face(avatar_ref_path)
        avatar_ref = self.transform(image=avatar_ref)["image"].unsqueeze(0)

        data = {
            "avatar_ref": avatar_ref,
            "user_a": user_a,
            "avatar_a": avatar_a,
            "user_frame": user_frames
        }
        
        return data


class InferenceAgent:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.rank = opt.rank

        self.init_network()
        self.data_processor = DataProcessor(opt)
        
    def init_network(self):
        from models.avatarforcing.AvatarForcing import AvatarForcing

        self.G = AvatarForcing(self.opt).to(self.rank)
        self.load_mae_ckpt(self.opt.mae_ckpt_path, rank=self.rank)
        self.load_ckpt(self.opt.ckpt_path, rank = self.rank)
        self.G.eval()
        # Inference-only — kill requires_grad so dynamo/inductor doesn't try
        # to trace a backward for ops like _scaled_mm (which has no autograd).
        for p in self.G.parameters():
            p.requires_grad_(False)

        # Optional FP8 quantization for Blackwell/Hopper tensor cores.
        # Dynamic (per-tensor-dynamic scaling) — no calibration needed, no
        # checkpoint changes, can be disabled by setting AF_FP8=0. Applied
        # BEFORE torch.compile so the compiler sees the quantized ops.
        if os.environ.get("AF_FP8", "0") == "1":
            try:
                from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
                # _scaled_mm on Blackwell requires bias in bf16/fp16 — cast
                # the module first so torchao's replacement finds bf16 biases.
                # Skipping the motion_autoencoder decoder: it uses
                # torch.linalg.qr which has no bf16 kernel. Flow_transformer
                # is the bigger cost anyway so this is still a big win.
                self.G.flow_transformer.to(torch.bfloat16)
                cfg = Float8DynamicActivationFloat8WeightConfig()
                print("> FP8 quantize: flow_transformer")
                quantize_(self.G.flow_transformer, cfg)
            except Exception as e:
                print(f"> FP8 quantize failed, falling back to BF16: {e}")

        if getattr(self.opt, "compile", False):
            # AF_COMPILE_MODE: "default" (baseline) or "reduce-overhead" (CUDA graphs).
            # reduce-overhead wraps compiled regions in CUDA graphs, which cuts
            # kernel-launch overhead but requires static shapes per graph.
            compile_mode = os.environ.get("AF_COMPILE_MODE", "reduce-overhead")
            print(f"> torch.compile: wrapping flow_transformer.forward_with_precomputed and motion_autoencoder.dec (mode={compile_mode})")
            self.G.flow_transformer.forward_with_precomputed = torch.compile(
                self.G.flow_transformer.forward_with_precomputed,
                mode=compile_mode, dynamic=True, fullgraph=False)
            self.G.motion_autoencoder.dec = torch.compile(
                self.G.motion_autoencoder.dec,
                mode=compile_mode, dynamic=False, fullgraph=False)

    def load_ckpt(self, ckpt_path, rank):
        state_dict = torch.load(ckpt_path, map_location='cuda:{}'.format(rank), weights_only=True)
        with torch.no_grad():
            for model_name, model_param in self.G.named_parameters():
                if model_name in state_dict:
                    model_param.copy_(state_dict[model_name].to(rank))
            print(f'> Loaded Avatar Forcing from: {ckpt_path}')

    def load_mae_ckpt(self, mae_ckpt_path = None, rank = None):
        state_dict = torch.load(mae_ckpt_path, map_location='cuda:{}'.format(rank), weights_only=True)
        with torch.no_grad():
            for model_name, param in self.G.named_parameters():
                if model_name in state_dict:
                    param.copy_(state_dict[model_name].to(rank))
            print(f"> Loaded Motion Latent Autoencoder from: {mae_ckpt_path}")

    @torch.no_grad()
    def run_stream(
        self,
        avatar_ref_path: str,
        avatar_audio_path: str,
        user_audio_path: str,
        user_video_path: str,
        a_cfg_scale: float    = 2.0,
        u_cfg_scale: float    = 1.0,
        nfe: int              = 10,
        seed: int             = 25
    ) -> None:
        """Streaming path: emits blocks as they are generated, pipes to ffmpeg."""
        t0 = time.perf_counter()
        data = self.data_processor.preprocess(
                avatar_ref_path   = avatar_ref_path,
                avatar_audio_path = avatar_audio_path,
                user_audio_path   = user_audio_path,
                user_video_path   = user_video_path
        )
        torch.cuda.synchronize()
        t_pre = time.perf_counter() - t0

        avatar_name = os.path.basename(avatar_ref_path).split(".")[0]
        res_video_path = os.path.join(self.opt.result_dir, f"{avatar_name}-stream-seed{seed}-{uuid.uuid4().hex[:10]}.mp4")

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tv:
            temp_filename = tv.name
        W = H = self.opt.input_size
        fps = self.opt.fps
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             temp_filename],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        block_latencies = []
        first_block_lat = None
        t_gen_start = time.perf_counter()
        total_frames = 0

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for block_idx, pixels, is_last in self.G.inference_stream(
                data         = data,
                a_cfg_scale  = a_cfg_scale,
                u_cfg_scale  = u_cfg_scale,
                nfe          = nfe,
                seed         = seed,
            ):
                torch.cuda.synchronize()
                t_now = time.perf_counter()
                if first_block_lat is None:
                    first_block_lat = t_now - t_gen_start
                block_latencies.append(t_now)

                # [1, k, 3, H, W] -> [k, H, W, 3] uint8 raw rgb
                vid = pixels.squeeze(0).detach().clamp(-1, 1).cpu()
                vid = ((vid - vid.min()) / (vid.max() - vid.min() + 1e-8) * 255.0)
                vid = vid.permute(0, 2, 3, 1).contiguous().to(torch.uint8)
                ff.stdin.write(vid.numpy().tobytes())
                total_frames += vid.shape[0]

        ff.stdin.close()
        ff.wait()
        t_gen = time.perf_counter() - t_gen_start

        # Mux audio
        with open(os.devnull, 'wb') as devnull:
            subprocess.call(
                ["ffmpeg", "-y", "-i", temp_filename, "-i", avatar_audio_path,
                 "-shortest", "-vcodec", "copy", "-acodec", "aac",
                 res_video_path],
                stdout=devnull, stderr=devnull)
            if os.path.exists(res_video_path):
                os.remove(temp_filename)

        # Per-block deltas (block N+1 time - block N time)
        if len(block_latencies) >= 2:
            per_block_ms = [(block_latencies[i] - block_latencies[i - 1]) * 1000 for i in range(1, len(block_latencies))]
            med = sorted(per_block_ms)[len(per_block_ms) // 2]
            mn, mx = min(per_block_ms), max(per_block_ms)
        else:
            med = mn = mx = 0.0

        clip_sec = total_frames / float(fps)
        rtf = t_gen / clip_sec if clip_sec > 0 else 0.0
        print(f"[stream] preprocess={t_pre:.2f}s  generate={t_gen:.2f}s  clip={clip_sec:.2f}s  RTF={rtf:.2f}x  frames={total_frames}")
        print(f"[stream] first-block-latency={first_block_lat*1000:.1f}ms  per-block median/min/max = {med:.1f} / {mn:.1f} / {mx:.1f} ms")
        print(f"[stream] output: {res_video_path}")

    @torch.no_grad()
    def run_inference(
        self,
        avatar_ref_path: str,
        avatar_audio_path: str,
        user_audio_path: str,
        user_video_path: str,
        a_cfg_scale: float    = 2.0,
        u_cfg_scale: float    = 1.0,
        nfe: int              = 10,
        seed: int             = 25
    ) -> None:

        t0 = time.perf_counter()
        data = self.data_processor.preprocess(
                avatar_ref_path   = avatar_ref_path,
                avatar_audio_path = avatar_audio_path,
                user_audio_path   = user_audio_path,
                user_video_path   = user_video_path
        )
        torch.cuda.synchronize()
        t_pre = time.perf_counter() - t0

        t1 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            d_hat = self.G.inference(
                data          = data,
                a_cfg_scale   = a_cfg_scale,
                u_cfg_scale   = u_cfg_scale,
                nfe           = nfe,
                seed          = seed,
                use_kv_cache  = True
            )['d_hat']
        torch.cuda.synchronize()
        t_gen = time.perf_counter() - t1

        t2 = time.perf_counter()
        avatar_name = os.path.basename(avatar_ref_path).split(".")[0]
        res_video_path = os.path.join(self.opt.result_dir, f"{avatar_name}-seed{seed}-{uuid.uuid4().hex[:10]}.mp4")
        self.save_video(d_hat, res_video_path, avatar_audio_path)
        t_mux = time.perf_counter() - t2

        clip_sec = float(d_hat.shape[0]) / float(self.opt.fps)
        rtf = t_gen / clip_sec if clip_sec > 0 else 0.0
        print(f"[timing] preprocess={t_pre:.2f}s  generate={t_gen:.2f}s  save={t_mux:.2f}s  "
              f"clip={clip_sec:.2f}s  RTF={rtf:.2f}x  (frames={d_hat.shape[0]}, nfe={nfe}, compile={getattr(self.opt, 'compile', False)})")

    def save_video(self, vid_target_recon:torch.Tensor, video_path:str, audio_path:str) -> str:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete = False) as temp_video:
            temp_filename = temp_video.name
            vid = vid_target_recon.permute(0, 2, 3, 1)
            vid = vid.detach().clamp(-1, 1).cpu()
            vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')
            T, H, W, C = vid.shape
            ff = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{W}x{H}", "-r", str(self.opt.fps),
                 "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 temp_filename],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ff.stdin.write(vid.numpy().tobytes())
            ff.stdin.close()
            ff.wait()
            
            if audio_path is not None: # add audio to video
                with open(os.devnull, 'wb') as f:
                    command =  "ffmpeg -y -i {} -i {} -shortest -vcodec h264 -acodec mp2 {}".format(temp_filename, audio_path, video_path)
                    subprocess.call(command, shell=True, stdout=f, stderr=f)

                if os.path.exists(video_path):
                    os.remove(temp_filename)
            else:
                os.rename(temp_filename, video_path)
            return video_path


def inference_args():
    parser = argparse.ArgumentParser(description='argument for avatar forcing')
    parser.add_argument('--infer_config', type=str,
            default = 'configs/inference.yaml', help='Path to the inference configuration file')
    parser.add_argument('--mae_ckpt_path', type=str,
            default = 'pretrained_dir/motion_autoencoder.pth', help='Version of the dataset to use')
    parser.add_argument('--ckpt_path',
            default = 'pretrained_dir/flow_transformer.pth', type = str, help = 'checkpoint path')

    # checkpoint
    parser.add_argument("--avatar_ref_path", type = str,
            default = None, help = 'path to reference image path')
    parser.add_argument("--avatar_audio_path", type = str,
            default = None, help = 'path to avatar audio')
    parser.add_argument("--user_audio_path", type = str,
            default = None, help = 'path to user audio path')
    parser.add_argument("--user_video_path", type = str,
            default = None, help = 'user video_path')
    parser.add_argument('--result_dir',
            default = "results", type = str, help = 'result dir')
    parser.add_argument('--res_video_path', type = str,
            default = None,  help = 'result video path')
    parser.add_argument('--seed', type=int,
            default=20)

    # NFE
    parser.add_argument('--nfe', type=int, default=10)

    # guidance scales
    parser.add_argument('--a_cfg_scale', type=float, default = 2)
    parser.add_argument('--u_cfg_scale', type=float, default = 1)

    # optimizations
    parser.add_argument('--compile', action='store_true',
            help='Enable torch.compile on flow_transformer and motion_autoencoder.dec')
    parser.add_argument('--warmup', type=int, default=0,
            help='Number of warmup runs before the timed run (useful with --compile)')
    parser.add_argument('--stream', action='store_true',
            help='Use streaming (block-by-block) path (Phase A).')

    return parser.parse_args()


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    opt = inference_args()
    seed_everything(opt.seed)
    
    infer_config = OmegaConf.load(opt.infer_config)
    opt = OmegaConf.create(vars(opt))
    opt = OmegaConf.merge(infer_config, opt)
    opt.rank, opt.ngpus  = 0, 1

    # initialize Inference Agent
    agent = InferenceAgent(opt)
    os.makedirs(opt.result_dir, exist_ok = True)

    run_fn = agent.run_stream if getattr(opt, "stream", False) else agent.run_inference

    for w in range(getattr(opt, "warmup", 0)):
        print(f"> warmup run {w+1}/{opt.warmup}")
        run_fn(
            avatar_ref_path   = opt.avatar_ref_path,
            avatar_audio_path = opt.avatar_audio_path,
            user_audio_path   = opt.user_audio_path,
            user_video_path   = opt.user_video_path,
            a_cfg_scale       = opt.a_cfg_scale,
            u_cfg_scale       = opt.u_cfg_scale,
            nfe               = opt.nfe,
        )

    run_fn(
        avatar_ref_path   = opt.avatar_ref_path,
        avatar_audio_path = opt.avatar_audio_path,
        user_audio_path   = opt.user_audio_path,
        user_video_path   = opt.user_video_path,

        a_cfg_scale       = opt.a_cfg_scale,
        u_cfg_scale       = opt.u_cfg_scale,

        nfe               = opt.nfe
    )
