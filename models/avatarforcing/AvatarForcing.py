import torch, math, os
import numpy as np
import torch.nn as nn

import torch.nn.functional as F
import torchvision.transforms.functional as TF

from tqdm import tqdm
from PIL import ImageDraw, ImageFont

from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput

from models import BaseModel, BaseDiffusionModel, FlowMatchScheduler
from models.wav2vec2 import Wav2VecModel

from models.avatarforcing.generator import Generator as MotionAutoencoder
from models.avatarforcing.flow_transformer import FlowTransformer


################ Encoders ################
class AudioEncoder(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        # initialize audio encoder
        self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only=True)
        self.wav2vec2.feature_extractor._freeze_parameters()
        for name, param in self.wav2vec2.named_parameters():
            param.requires_grad = False

        # set num of frames
        self.only_last_features = opt.only_last_features        
        self.num_frames_for_clip = int(opt.sec * self.opt.fps)
        self.samples_per_frame = int(opt.sampling_rate / opt.fps)

        # set audio projection layers
        audio_input_dim = 768 if opt.only_last_features else 12 * 768
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_input_dim, opt.dim_w),
            nn.LayerNorm(opt.dim_w),
            nn.SiLU())

    def get_wav2vec2_feature(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
        a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states = not self.only_last_features)
        if self.only_last_features:
            a = a.last_hidden_state
        else:
            a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
            a = a.reshape(a.shape[0], a.shape[1], -1)
        return a

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if a.shape[1] % int(self.num_frames_for_clip * self.samples_per_frame) != 0:
            a = F.pad(a, (0, int(self.num_frames_for_clip * self.samples_per_frame) - a.shape[1]), mode='replicate')
        a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip)
        return self.audio_projection(a)

    @torch.inference_mode()
    def inference(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
        if a.shape[1] % int(seq_len * self.samples_per_frame) != 0:
            a = F.pad(a, (0, int(seq_len * self.samples_per_frame) - a.shape[1]), mode='replicate')
        a = self.get_wav2vec2_feature(a, seq_len=seq_len)
        return self.audio_projection(a)

    @torch.inference_mode()
    def rolling_slice(self, a: torch.Tensor, frame_lo: int, frame_hi: int,
                      lookback_frames: int = 20, lookahead_frames: int = 0) -> torch.Tensor:
        """Compute audio features for video-frames [frame_lo, frame_hi) via a windowed wav2vec.

        Drop-in replacement for ``full_batch_feats[:, frame_lo:frame_hi]`` where
        ``full_batch_feats = self.inference(full_audio, seq_len=T)``. Instead of
        running wav2vec on the full audio track, run it on a narrow window
        [(frame_lo - lookback_frames)*spf, (frame_hi + lookahead_frames)*spf]
        and slice the requested frames out.

        For true streaming use ``lookahead_frames=0``. The lookback provides
        wav2vec context for the current frames.
        """
        want_lo = max(0, frame_lo - lookback_frames)
        want_hi = frame_hi + lookahead_frames
        start = want_lo * self.samples_per_frame
        end   = want_hi * self.samples_per_frame
        window_audio = a[:, start:end]
        window_frames = want_hi - want_lo
        if window_frames <= 0:
            return torch.zeros(a.shape[0], 0, self.opt.dim_w, dtype=a.dtype, device=a.device)
        feats = self.inference(window_audio, seq_len=window_frames)
        # Slice out the requested sub-range. If we had to clip lookback at 0,
        # the requested window starts at (frame_lo - want_lo) inside feats.
        out_lo = frame_lo - want_lo
        out_hi = out_lo + (frame_hi - frame_lo)
        return feats[:, out_lo:out_hi]



######## Main Model ########
class AvatarForcing(BaseDiffusionModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.rank = opt.rank
        
        self.fps = opt.fps
        self.sampling_rate = opt.sampling_rate
        self.samples_per_frame = int(self.sampling_rate / self.fps)
        self.num_frames_for_clip = int(self.opt.sec * self.fps)

        # motion autoencoder
        self.motion_autoencoder = MotionAutoencoder(size = opt.input_size, style_dim = opt.dim_w, motion_dim = opt.dim_m)
        self.motion_autoencoder.requires_grad_(False)
        self.num_frames_per_block = opt.num_frames_per_block

        # condition encoders
        self.audio_encoder = AudioEncoder(opt)

        # Flow Models
        self.num_train_timestep = opt.num_train_timestep
        self.min_step = int(0 * self.num_train_timestep)
        self.max_step = int(1 * self.num_train_timestep)
        self.flow_transformer = FlowTransformer(opt)

        self.scheduler.timesteps = self.scheduler.timesteps.to(self.rank)


    ############### Set model for training ##############
    def set_requires_grad(self, requires_grad: bool) -> None:
        for name, param in self.named_parameters():
            if "motion_autoencoder" in name or 'wav2vec2' in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(requires_grad)

    def get_parameters_for_train(self) -> list:
        return [v for k, v in self.named_parameters() if v.requires_grad]

    #####################################################
    
    @torch.inference_mode()
    def encode_user_motion(self, user_frame_list):
        outputs = []
        for i in range(0, len(user_frame_list), self.num_frames_for_clip):
            batch_frames = user_frame_list[i:i + self.num_frames_for_clip]
            user_frame_tensor = torch.cat(batch_frames, dim=0).to(self.rank)
            user_r_d = self.motion_autoencoder.enc.enc_motion(user_frame_tensor)
            user_r_d = self.motion_autoencoder.dec.direction(user_r_d)
            outputs.append(user_r_d)
        user_r_d = torch.cat(outputs, dim=0).unsqueeze(0)
        return user_r_d


    @torch.inference_mode()
    def decode_latent_into_image(
        self, s_r: torch.Tensor, r_s: torch.Tensor,
        s_r_feats: list, r_d: torch.Tensor
    ) -> dict:

        T = r_d.shape[1]
        B = r_d.shape[0]
        block_size = self.num_frames_per_block
        d_hat_blocks = []
        s_r = s_r.unsqueeze(1)
        
        s_r_feats_expanded = [f.repeat_interleave(block_size, dim=0) for f in s_r_feats]
        
        for block_idx in range(0, T, block_size):
            block_end = min(block_idx + block_size, T)
            block_len = block_end - block_idx

            r_d_block = r_d[:, block_idx:block_end]  # [B, block_len, D]
            
            needs_padding = block_len < block_size
            if needs_padding:
                pad_len = block_size - block_len
                r_d_block = F.pad(r_d_block, (0, 0, 0, pad_len), mode='replicate')
            
            s_r_d_block = s_r + r_d_block
            s_r_d_block = s_r_d_block.reshape(B * block_size, -1)

            feats_for_block = s_r_feats_expanded
            img_block, _ = self.motion_autoencoder.dec(s_r_d_block, alpha=None, feats=feats_for_block)            
            img_block = img_block.reshape(B, block_size, *img_block.shape[1:])
            
            if needs_padding: img_block = img_block[:, :block_len]
            d_hat_blocks.append(img_block)

        d_hat = torch.cat(d_hat_blocks, dim=1).squeeze()

        return {'d_hat': d_hat}


    #####################################################

    @torch.inference_mode()
    def update_kv_cache(
        self,
        final_latents: torch.Tensor,

        precomputed_c,
        precomputed_wr,
        precomputed_adaLN,  # not used — KV cache update always uses t=0

        start_pos: int = 0
    ):
        x_cat = torch.cat([final_latents, final_latents], dim=0)
        B, L = x_cat.shape[0:2]
        t_zero = torch.zeros(B, L, dtype=torch.float32, device=final_latents.device)
        t_zero_embed = self.flow_transformer.t_embedder(t_zero.flatten()).unflatten(0, t_zero.shape)

        shared_scale_shift_alpha = self.flow_transformer.adaLN_modulation(t_zero_embed).chunk(6, dim=-1)
        final_adaLN = self.flow_transformer.final_layer.adaLN_modulation(t_zero_embed).chunk(2, dim=-1)
        adaLN_t0 = (shared_scale_shift_alpha, t_zero_embed, final_adaLN)

        N = x_cat.shape[1]
        freqs_cis = self.flow_transformer.blocks[0].attn.freqs_cis
        freqs_cis_slice = freqs_cis.narrow(0, int(start_pos), N)
        self.flow_transformer.forward_with_precomputed(
            x                 = x_cat,
            precomputed_c     = precomputed_c,
            precomputed_adaLN = adaLN_t0,
            precomputed_wr    = precomputed_wr,
            freqs_cis_slice   = freqs_cis_slice,
            kv_cache_list     = self.kv_cache,
            update_kv_cache   = True)


    def initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize KV cache with shape (B*2, H, L, D) for 2-branch CFG
        (uncond + audio-conditioned). The 3rd user-motion branch was dropped
        because this app zeroes user_motion/user_wa entirely."""
        self.cache_len = self.num_frames_for_clip - self.num_frames_per_block
        self.cross_cache_len = self.opt.attention_window  # 2
        kv_cache = []
        head_dim = self.opt.dim_h // self.opt.num_heads
        for _ in range(self.opt.transformer_depth):
            self_kv_cache_dict = {
                "k": torch.zeros((batch_size * 2, self.opt.num_heads, self.cache_len - 2, head_dim), dtype=dtype, device=device),
                "v": torch.zeros((batch_size * 2, self.opt.num_heads, self.cache_len - 2, head_dim), dtype=dtype, device=device)}
            cross_kv_cache_dict = {
                "k": torch.zeros((batch_size * 2, self.opt.num_heads, self.cross_cache_len, head_dim), dtype=dtype, device=device),
                "v": torch.zeros((batch_size * 2, self.opt.num_heads, self.cross_cache_len, head_dim), dtype=dtype, device=device)}
            kv_cache.append((self_kv_cache_dict, cross_kv_cache_dict))
        self.kv_cache = kv_cache # list


    @torch.inference_mode()
    def inference(
        self,
        data: dict,
        a_cfg_scale = None,
        u_cfg_scale = None,
        nfe         = 10,
        seed        = None,
        use_kv_cache: bool = True,
    ) -> dict:
        import time as _time
        _prof = os.environ.get("AF_PROFILE", "0") == "1"
        def _t(label, t0):
            if _prof:
                torch.cuda.synchronize()
                print(f"  [prof] {label:<40s} {(_time.perf_counter()-t0)*1000:8.1f} ms")

        t0 = _time.perf_counter()
        s = data['avatar_ref']
        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))
        _t("encode_image_into_latent (ref)", t0)
        if 's_r' in data:
            r_s = self.encode_identity_into_motion(s_r)
        else:
            r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
        data['r_s'] = r_s

        t1 = _time.perf_counter()
        sample = self.sample(
            data          = data,
            a_cfg_scale   = a_cfg_scale,
            u_cfg_scale   = u_cfg_scale,
            nfe           = nfe,
            seed          = seed,
            use_kv_cache  = use_kv_cache,
        )
        _t("sample() [diffusion + audio + user_motion enc]", t1)

        t2 = _time.perf_counter()
        data_out = self.decode_latent_into_image(
            s_r       = s_r,
            r_s       = r_s,
            s_r_feats = s_r_feats,
            r_d       = sample
        )
        _t("decode_latent_into_image (pixel decode)", t2)

        return data_out

    @torch.inference_mode()
    def sample(
        self,
        data: dict         = {},
        a_cfg_scale: float = 1.0,
        u_cfg_scale: float = 1.0,
        nfe: int           = 10,
        seed: int          = None,
        use_kv_cache: bool = True
    ) -> torch.Tensor:
        r_s, avatar_a, user_a, user_frame = data['r_s'], data['avatar_a'], data['user_a'], data["user_frame"]

        import time as _time
        _prof = os.environ.get("AF_PROFILE", "0") == "1"
        def _t(label, t0):
            if _prof:
                torch.cuda.synchronize()
                print(f"    [prof] {label:<38s} {(_time.perf_counter()-t0)*1000:8.1f} ms")

        B = r_s.shape[0]
        T = math.ceil(max(avatar_a.shape[-1], user_a.shape[-1]) * self.fps / self.sampling_rate)

        # Basic setups
        self.denoising_step_list = torch.tensor(np.linspace(self.opt.num_train_timestep, 0, nfe - 1).tolist())
        self.initialize_kv_cache(batch_size=B, dtype=avatar_a.dtype, device=self.rank)

        _a = _time.perf_counter()
        avatar_a, user_a = avatar_a.to(self.rank), user_a.to(self.rank)
        avatar_wa = self.audio_encoder.inference(avatar_a, seq_len=T)
        user_wa  = self.audio_encoder.inference(user_a, seq_len=T)
        _t("audio_encoder.inference x2 (wav2vec)", _a)

        # Computing the first block
        avatar_wa_t = avatar_wa[:, :self.num_frames_for_clip]
        if avatar_wa_t.shape[1] < self.num_frames_for_clip:
            avatar_wa_t = F.pad(avatar_wa_t, (0, 0, 0, self.num_frames_for_clip - avatar_wa_t.shape[1]), mode='replicate')

        user_wa_t = user_wa[:, :self.num_frames_for_clip]
        if user_wa_t.shape[1] < self.num_frames_for_clip:
            user_wa_t = F.pad(user_wa_t, (0, 0, 0, self.num_frames_for_clip - user_wa_t.shape[1]), mode='replicate')

        _b = _time.perf_counter()
        user_r_d = self.encode_user_motion(user_frame)
        _t("encode_user_motion (all user frames)", _b)

        _c = _time.perf_counter()
        user_r_d_t = user_r_d[:, :self.num_frames_for_clip]

        if user_r_d_t.shape[1] < self.num_frames_for_clip:
            user_r_d_t = F.pad(user_r_d_t, (0, 0, 0, self.num_frames_for_clip - user_r_d_t.shape[1]), mode='replicate')        
        x_t = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=self.rank)
        precomputed_c, precomputed_wr, precomputed_adaLN = self.prepare_cfg_condition(
            avatar_wa_t, user_wa_t, user_r_d_t, r_s, seq_len=self.num_frames_for_clip, context_len=0)

        samples = []
        start_pos = 0

        for index, current_timestep in enumerate(self.denoising_step_list):
            is_final_step = (index == len(self.denoising_step_list) - 1)
            x_t = self.solve_cfg(
                B                = B,
                index            = index,
                current_timestep = current_timestep,
                x_t              = x_t,

                precomputed_c     = precomputed_c,
                precomputed_wr    = precomputed_wr,
                precomputed_adaLN = precomputed_adaLN,

                start_pos       = start_pos,
                context_len     = 0,

                use_kv_cache    = False,
                a_cfg_scale     = a_cfg_scale,
                u_cfg_scale     = u_cfg_scale,
            )

            if is_final_step:        
                self.update_kv_cache(
                    final_latents     = x_t,
                    precomputed_c     = precomputed_c,
                    precomputed_wr    = precomputed_wr,
                    precomputed_adaLN = precomputed_adaLN,
                    start_pos         = start_pos
                )

        samples.append(x_t)

        # Computing the subsequent blocks
        if T - self.num_frames_for_clip > 0:
            for t in range(self.num_frames_for_clip, T, self.num_frames_per_block):
                start_pos = t - 2 if use_kv_cache else t - (self.num_frames_for_clip - self.num_frames_per_block)

                ss_idx, ee_idx = t, t + self.num_frames_per_block
                s_idx, e_idx   = ss_idx * self.samples_per_frame, ee_idx * self.samples_per_frame

                avatar_wa_t = avatar_wa[:, ss_idx - 2: ee_idx]
                if avatar_wa_t.shape[1] < self.num_frames_per_block + 2:
                    avatar_wa_t = F.pad(avatar_wa_t, (0, 0, 0, self.num_frames_per_block + 2 - avatar_wa_t.shape[1]), mode='replicate')

                user_wa_t = user_wa[:, ss_idx - 2: ee_idx]
                if user_wa_t.shape[1] < self.num_frames_per_block + 2:
                    user_wa_t = F.pad(user_wa_t, (0, 0, 0, self.num_frames_per_block + 2 - user_wa_t.shape[1]), mode='replicate')

                user_r_d_t = user_r_d[:, ss_idx-2: ee_idx].to(self.rank)
                if user_r_d_t.shape[1] < self.num_frames_per_block + 2:
                    user_r_d_t = F.pad(user_r_d_t, (0, 0, 0, self.num_frames_per_block + 2 - user_r_d_t.shape[1]), mode='replicate')

                # initialize noise block
                offset_x_t = x_t[:, -2:]
                noise_t = torch.randn(B, self.num_frames_per_block, self.opt.dim_w, device=self.rank)
                x_t = torch.cat([offset_x_t, noise_t], dim=1)

                precomputed_c, precomputed_wr, precomputed_adaLN = self.prepare_cfg_condition(
                    avatar_wa_t, user_wa_t, user_r_d_t, r_s, seq_len = self.num_frames_per_block, context_len = 2)

                for index, current_timestep in enumerate(self.denoising_step_list):
                    is_final_step = (index == len(self.denoising_step_list) - 1)

                    x_t = self.solve_cfg(
                        B                = B,
                        index            = index,
                        current_timestep = current_timestep,
                        x_t              = x_t,

                        precomputed_c     = precomputed_c,
                        precomputed_wr    = precomputed_wr,
                        precomputed_adaLN = precomputed_adaLN,

                        start_pos       = start_pos,
                        context_len     = 2,
                        use_kv_cache    = use_kv_cache,

                        a_cfg_scale     = a_cfg_scale,
                        u_cfg_scale     = u_cfg_scale
                    )
                    
                    if is_final_step:
                        self.update_kv_cache(
                            final_latents     = x_t,

                            precomputed_c     = precomputed_c,
                            precomputed_wr    = precomputed_wr,
                            precomputed_adaLN = precomputed_adaLN,

                            start_pos         = start_pos
                        )

                samples.append(x_t[:, -self.num_frames_per_block:])
        
        _t("flow_transformer total (all blocks x NFE)", _c)
        samples = torch.cat(samples, dim=1)[:, :T]
        return samples


    # ---------------- Streaming inference path ----------------
    @torch.inference_mode()
    def _decode_one_block(self, latent_block: torch.Tensor, s_r_unsq: torch.Tensor,
                          s_r_feats_expanded: list) -> torch.Tensor:
        """Decode a single latent block [1, k, D] to pixels [1, k, 3, H, W] (k <= num_frames_per_block)."""
        B = latent_block.shape[0]
        block_size = self.num_frames_per_block
        block_len = latent_block.shape[1]
        needs_padding = block_len < block_size
        if needs_padding:
            latent_block = F.pad(latent_block, (0, 0, 0, block_size - block_len), mode='replicate')
        s_r_d = s_r_unsq + latent_block
        s_r_d = s_r_d.reshape(B * block_size, -1)
        img, _ = self.motion_autoencoder.dec(s_r_d, alpha=None, feats=s_r_feats_expanded)
        img = img.reshape(B, block_size, *img.shape[1:])
        if needs_padding:
            img = img[:, :block_len]
        return img

    @torch.inference_mode()
    def inference_stream(
        self,
        data: dict,
        a_cfg_scale: float = 2.0,
        u_cfg_scale: float = 1.0,
        nfe: int           = 10,
        seed: int          = 25,
        avatar_wa: "torch.Tensor | None" = None,
        user_wa:   "torch.Tensor | None" = None,
    ):
        """
        Generator yielding per-block decoded pixel tensors.
        Yields tuples (block_idx:int, pixels:[1, k, 3, H, W], is_last:bool).

        If `avatar_wa` / `user_wa` are provided, they are used directly
        (expected shape [1, T, dim_w] already through self.audio_projection).
        If None, they are computed internally from `data['avatar_a']` and
        `data['user_a']` — original behavior.
        """
        import time as _time
        _prof = os.environ.get("AF_PROFILE", "0") == "1"

        # ---- Reference portrait: encode once ----
        s = data['avatar_ref'].to(self.opt.rank)
        s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s)
        r_s = self.motion_autoencoder.dec.direction(r_s_lambda)

        # Pre-expand s_r_feats for the fixed block_size we'll pass to the decoder
        block_size = self.num_frames_per_block
        s_r_feats_expanded = [f.repeat_interleave(block_size, dim=0) for f in s_r_feats]
        s_r_unsq = s_r.unsqueeze(1)  # [1, 1, D]

        # ---- Inputs ----
        avatar_a = data['avatar_a'].to(self.rank)
        user_a   = data['user_a'].to(self.rank)
        user_frame_list = data['user_frame']   # list of [1, 3, H, W] tensors

        B = 1
        T = math.ceil(max(avatar_a.shape[-1], user_a.shape[-1]) * self.fps / self.sampling_rate)
        T = min(T, len(user_frame_list))

        self.denoising_step_list = torch.tensor(np.linspace(self.opt.num_train_timestep, 0, nfe - 1).tolist())
        self.initialize_kv_cache(batch_size=B, dtype=avatar_a.dtype, device=self.rank)

        # Audio features: use caller-provided tensors if available, otherwise
        # compute from raw audio (original path). The "provided" path is what
        # the voice-chat session uses: it batches wav2vec at phrase boundaries
        # and feeds the resulting features here.
        if avatar_wa is None:
            avatar_wa = self.audio_encoder.inference(avatar_a, seq_len=T)
        if user_wa is None:
            user_wa   = self.audio_encoder.inference(user_a,   seq_len=T)

        # ---- First block (prime, 50 frames, no KV cache) ----
        seq0 = self.num_frames_for_clip
        avatar_wa_t = avatar_wa[:, :seq0]
        if avatar_wa_t.shape[1] < seq0:
            avatar_wa_t = F.pad(avatar_wa_t, (0, 0, 0, seq0 - avatar_wa_t.shape[1]), mode='replicate')
        user_wa_t = user_wa[:, :seq0]
        if user_wa_t.shape[1] < seq0:
            user_wa_t = F.pad(user_wa_t, (0, 0, 0, seq0 - user_wa_t.shape[1]), mode='replicate')

        user_r_d_0 = self.encode_user_motion(user_frame_list[:seq0])  # [1, seq0, D]
        user_r_d_t = user_r_d_0[:, :seq0]
        if user_r_d_t.shape[1] < seq0:
            user_r_d_t = F.pad(user_r_d_t, (0, 0, 0, seq0 - user_r_d_t.shape[1]), mode='replicate')

        x_t = torch.randn(B, seq0, self.opt.dim_w, device=self.rank)
        precomputed_c, precomputed_wr, precomputed_adaLN = self.prepare_cfg_condition(
            avatar_wa_t, user_wa_t, user_r_d_t, r_s, seq_len=seq0, context_len=0)

        for index, current_timestep in enumerate(self.denoising_step_list):
            is_final_step = (index == len(self.denoising_step_list) - 1)
            x_t = self.solve_cfg(
                B=B, index=index, current_timestep=current_timestep, x_t=x_t,
                precomputed_c=precomputed_c, precomputed_wr=precomputed_wr,
                precomputed_adaLN=precomputed_adaLN,
                start_pos=0, context_len=0,
                use_kv_cache=False,
                a_cfg_scale=a_cfg_scale, u_cfg_scale=u_cfg_scale,
            )
            if is_final_step:
                self.update_kv_cache(
                    final_latents=x_t, precomputed_c=precomputed_c,
                    precomputed_wr=precomputed_wr, precomputed_adaLN=precomputed_adaLN,
                    start_pos=0)

        # Decode the primer in block_size chunks
        for bstart in range(0, seq0, block_size):
            bend = min(bstart + block_size, seq0)
            latent_block = x_t[:, bstart:bend]
            is_last_of_prime = (bend >= seq0)
            pixels = self._decode_one_block(latent_block, s_r_unsq, s_r_feats_expanded)
            yield (bstart // block_size, pixels, False)

        prev_x_t_tail = x_t  # keep last 2 tokens as overlap into the next block

        # ---- Subsequent blocks ----
        block_idx_counter = seq0 // block_size
        for t_idx in range(seq0, T, block_size):
            tblk = _time.perf_counter()
            start_pos = t_idx - 2
            ss_idx, ee_idx = t_idx, t_idx + block_size

            avatar_wa_t = avatar_wa[:, ss_idx - 2: ee_idx]
            if avatar_wa_t.shape[1] < block_size + 2:
                avatar_wa_t = F.pad(avatar_wa_t, (0, 0, 0, block_size + 2 - avatar_wa_t.shape[1]), mode='replicate')
            user_wa_t = user_wa[:, ss_idx - 2: ee_idx]
            if user_wa_t.shape[1] < block_size + 2:
                user_wa_t = F.pad(user_wa_t, (0, 0, 0, block_size + 2 - user_wa_t.shape[1]), mode='replicate')

            frame_lo = max(0, ss_idx - 2)
            frame_hi = min(len(user_frame_list), ee_idx)
            user_r_d_window = self.encode_user_motion(user_frame_list[frame_lo:frame_hi])  # [1, (<=12), D]
            # Left-pad if we grabbed fewer than 2 history frames (only at start boundary)
            if frame_lo > ss_idx - 2:
                deficit = (ss_idx - 2) - frame_lo
                user_r_d_window = F.pad(user_r_d_window, (0, 0, deficit, 0), mode='replicate')
            user_r_d_t = user_r_d_window
            if user_r_d_t.shape[1] < block_size + 2:
                user_r_d_t = F.pad(user_r_d_t, (0, 0, 0, block_size + 2 - user_r_d_t.shape[1]), mode='replicate')

            offset_x_t = prev_x_t_tail[:, -2:]
            noise_t = torch.randn(B, block_size, self.opt.dim_w, device=self.rank)
            x_t = torch.cat([offset_x_t, noise_t], dim=1)

            precomputed_c, precomputed_wr, precomputed_adaLN = self.prepare_cfg_condition(
                avatar_wa_t, user_wa_t, user_r_d_t, r_s, seq_len=block_size, context_len=2)

            for index, current_timestep in enumerate(self.denoising_step_list):
                is_final_step = (index == len(self.denoising_step_list) - 1)
                x_t = self.solve_cfg(
                    B=B, index=index, current_timestep=current_timestep, x_t=x_t,
                    precomputed_c=precomputed_c, precomputed_wr=precomputed_wr,
                    precomputed_adaLN=precomputed_adaLN,
                    start_pos=start_pos, context_len=2,
                    use_kv_cache=True,
                    a_cfg_scale=a_cfg_scale, u_cfg_scale=u_cfg_scale,
                )
                if is_final_step:
                    self.update_kv_cache(
                        final_latents=x_t, precomputed_c=precomputed_c,
                        precomputed_wr=precomputed_wr, precomputed_adaLN=precomputed_adaLN,
                        start_pos=start_pos)

            # Decode this block's 10 new frames (excluding the 2 overlap)
            latent_block = x_t[:, -block_size:]
            # Clip to remaining true length (last block may be short)
            remaining = T - t_idx
            if remaining < block_size:
                latent_block = latent_block[:, :remaining]
            pixels = self._decode_one_block(latent_block, s_r_unsq, s_r_feats_expanded)

            prev_x_t_tail = x_t
            is_last = (t_idx + block_size >= T)
            if _prof:
                torch.cuda.synchronize()
                print(f"    [prof-stream] block {block_idx_counter:3d}  gen+dec={(_time.perf_counter()-tblk)*1000:7.1f} ms")
            yield (block_idx_counter, pixels, is_last)
            block_idx_counter += 1

    def solve_cfg(self, B: int, index: int, current_timestep: float, x_t: torch.Tensor,
        precomputed_c: torch.Tensor, precomputed_wr: torch.Tensor, precomputed_adaLN: torch.Tensor,
        start_pos: int, context_len: int, use_kv_cache: bool, a_cfg_scale: float, u_cfg_scale: float,
        update_kv_cache: bool = False):

        seq_len = self.num_frames_per_block if use_kv_cache else self.num_frames_for_clip
        x_cat = x_t.repeat(2, 1, 1)
        precomputed_adaLN = precomputed_adaLN[index] if precomputed_adaLN is not None else None

        N = x_cat.shape[1]
        freqs_cis = self.flow_transformer.blocks[0].attn.freqs_cis
        freqs_cis_slice = freqs_cis.narrow(0, int(start_pos), N)

        v = self.flow_transformer.forward_with_precomputed(
            x                       = x_cat,
            precomputed_c           = precomputed_c,
            precomputed_adaLN       = precomputed_adaLN,
            precomputed_wr          = precomputed_wr,

            freqs_cis_slice         = freqs_cis_slice,
            kv_cache_list           = self.kv_cache if use_kv_cache else None,
            update_kv_cache         = update_kv_cache)

        v_uncond, v_aud = v.chunk(2, dim=0)
        # user-motion branch dropped — this app doesn't use it. u_cfg_scale
        # is accepted but ignored for backward compat.
        v_t = v_uncond + a_cfg_scale * (v_aud - v_uncond)

        next_timestep = self.denoising_step_list[index + 1] if index < len(self.denoising_step_list) - 1 else current_timestep

        if use_kv_cache: # kv inference
            t_for_denoise_single = torch.cat([
                torch.zeros((B, 2)),
                torch.full((B, self.num_frames_per_block), current_timestep)], dim=1).to(dtype=torch.float32, device=self.rank)
            t_for_next_denoise = torch.cat([
                torch.zeros((B, 2)),
                torch.full((B, self.num_frames_per_block), next_timestep)], dim=1).to(dtype=torch.float32, device=self.rank)
        elif start_pos == 0: # first block (both kv and non-kv)
            t_for_denoise_single = torch.full((B, self.num_frames_for_clip), current_timestep).to(dtype=torch.float32, device=self.rank)
            t_for_next_denoise = torch.full((B, self.num_frames_for_clip), next_timestep).to(dtype=torch.float32, device=self.rank)
        else: # non-kv infernece
            t_for_denoise_single = torch.cat([
                torch.zeros((B, self.num_frames_for_clip - self.num_frames_per_block)),
                torch.full((B, self.num_frames_per_block), current_timestep)], dim=1).to(dtype=torch.float32, device=self.rank)
            t_for_next_denoise = torch.cat([
                torch.zeros((B, self.num_frames_for_clip - self.num_frames_per_block)),
                torch.full((B, self.num_frames_per_block), next_timestep)], dim=1).to(dtype=torch.float32, device=self.rank)
        return self.solve_closed(flow_pred=v_t, xt=x_t, timestep=t_for_denoise_single, next_timestep=t_for_next_denoise)
        

    def prepare_cfg_condition(self, wa, wa_user, motion_user, wr, seq_len, context_len=0) -> tuple:
        B = wa.shape[0]
        null_wa = torch.zeros_like(wa)
        null_user_wa = torch.zeros_like(wa_user)
        null_user_motion = torch.zeros_like(motion_user)

        # 2-branch CFG: [uncond, audio]. The 3rd user-motion branch was
        # dropped. user_wa/motion are always null now.
        wa_cat = torch.cat([null_wa, wa], dim=0)
        wa_user_cat = torch.cat([null_user_wa, null_user_wa], dim=0)
        motion_user_cat = torch.cat([null_user_motion, null_user_motion], dim=0)
        wr_cat = torch.cat([wr, wr], dim=0)
        
        precomputed_c = self.flow_transformer.compute_condition(wa = wa_cat, wa_user = wa_user_cat, motion_user = motion_user_cat, train = False)
        precomputed_wr = self.flow_transformer.compute_wr_embedded(wr = wr_cat, seq_len = seq_len + context_len, train = False)

        timestep_list = [t.item() for t in self.denoising_step_list]
        precomputed_adaLN = self.flow_transformer.precompute_timestep_adaLN(timestep_list=timestep_list, batch_size=B * 2, seq_len=seq_len, context_len=context_len, device=self.rank)
        return precomputed_c, precomputed_wr, precomputed_adaLN


    def solve_closed(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, next_timestep: torch.Tensor) -> torch.Tensor:
        orig_dtype, device = xt.dtype, xt.device

        v  = flow_pred.to(device=device, dtype=torch.double)  # [B,T,D]
        x  = xt.to(device=device, dtype=torch.double)         # [B,T,D]
        timesteps_tbl = self.scheduler.timesteps.to(device=device, dtype=torch.double)  # (K,)
        sigmas_tbl    = self.scheduler.sigmas.to(device=device, dtype=torch.double)     # (K,)

        t_cur  = timestep.to(device=device, dtype=torch.double)         # [B,T]
        t_next = next_timestep.to(device=device, dtype=torch.double)    # [B,T]

        diffs_t  = (timesteps_tbl.view(1,1,-1) - t_cur.unsqueeze(-1)).abs()   # [B,T,K]
        diffs_tp = (timesteps_tbl.view(1,1,-1) - t_next.unsqueeze(-1)).abs()  # [B,T,K]
        idx_t    = diffs_t.argmin(dim=-1)                                   # [B,T]
        idx_tp   = diffs_tp.argmin(dim=-1)                                  # [B,T]
        
        sigma_t  = sigmas_tbl[idx_t]     # [B,T]
        sigma_tp = sigmas_tbl[idx_tp]    # [B,T]

        zero_t  = (timestep == 0).to(device=device)
        zero_tp = (next_timestep == 0).to(device=device)
        sigma_t  = torch.where(zero_t,  torch.zeros_like(sigma_t),  sigma_t)
        sigma_tp = torch.where(zero_tp, torch.zeros_like(sigma_tp), sigma_tp)

        alpha_t, alpha_tp = 1.0 - sigma_t, 1.0 - sigma_tp

        x0_hat = x - sigma_t.unsqueeze(-1) * v   # [B,T,D]
        ratio = torch.zeros_like(sigma_tp)
        valid = (sigma_t > 0)
        ratio[valid] = sigma_tp[valid] / torch.clamp(sigma_t[valid], min=1e-8)
        xt_next = (ratio.unsqueeze(-1) * x + (alpha_tp - ratio * alpha_t).unsqueeze(-1) * x0_hat)
        hist_mask = (zero_t & zero_tp).unsqueeze(-1)  # [B,T,1]
        xt_next = torch.where(hist_mask, x, xt_next)
        return xt_next.to(orig_dtype)


    ######## Motion encoder - decoder ########
    @torch.inference_mode()
    def encode_image_into_latent(self, x: torch.Tensor) -> list:
        x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return x_r, x_r_lambda, x_r_feats


    @torch.inference_mode()
    def encode_identity_into_motion(self, x_r: torch.Tensor) -> torch.Tensor:
        if len(x_r.shape) == 3:
            b, t = x_r.shape[0:2]
            x_r = x_r.reshape(b*t, *x_r.shape[2:])
            x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
            r_x = self.motion_autoencoder.dec.direction(x_r_lambda).reshape(b, t, -1)
        else:
            x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
            r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
        return r_x


    @torch.inference_mode()
    def encode_identity_into_lambda(self, x_r: torch.Tensor) -> torch.Tensor:
        x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
        return x_r_lambda

    @torch.inference_mode()
    def decode_block(
        self,
        r_d_block: torch.Tensor,
        s_r: torch.Tensor,
        s_r_feats_expanded: list,
        block_size: int,
        B: int
    ) -> torch.Tensor:
        block_len = r_d_block.shape[1]
        
        s_r_d_block = s_r + r_d_block
        s_r_d_block = s_r_d_block.reshape(B * block_len, -1)

        img_block, _ = self.motion_autoencoder.dec(s_r_d_block, alpha=None, feats=s_r_feats_expanded)
        img_block = img_block.reshape(B, block_len, *img_block.shape[1:])
        return img_block.squeeze(0)

