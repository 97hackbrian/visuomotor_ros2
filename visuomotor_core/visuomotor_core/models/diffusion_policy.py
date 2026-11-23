from __future__ import annotations  # compatibilidad Python 3.8 (Jetson JetPack 5)

from collections import deque
import threading
import torch
import torchvision
from torch import Tensor, nn
import numpy as np
import math

import einops
from typing import Callable
import torch.nn.functional as F 
from huggingface_hub import PyTorchModelHubMixin
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


from visuomotor_core.models.config import DiffusionConfig
from visuomotor_core.dataset.normalize import Normalize, Unnormalize
from visuomotor_core.models.utils import populate_queues, get_device_from_parameters, get_dtype_from_parameters



# ----------------------------------------------------------------------------------------------
# Operaciones de cuaterniones en torch ([..., 4] = [qx, qy, qz, qw]); espejo de visuomotor_core/utils/rotations.py
# ----------------------------------------------------------------------------------------------
def _quat_normalize(q: Tensor) -> Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_multiply(a: Tensor, b: Tensor) -> Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def _quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def _quat_canonicalize(q: Tensor, mode: str, ref: Tensor | None) -> Tensor:
    # signo = -1 donde corresponde invertir (aritmética portable a torch 2.0 de la Jetson, sin torch.where escalar)
    if mode == "home" and ref is not None:
        flip = ((q * ref).sum(-1, keepdim=True) < 0).to(q.dtype)
    else:
        flip = (q[..., 3:4] < 0).to(q.dtype)
    return q * (1.0 - 2.0 * flip)


class DiffusionPolicy(nn.Module, PyTorchModelHubMixin):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = DiffusionConfig()
        elif isinstance(config, dict):
            # `from_pretrained` entrega el config.json como dict cuando las anotaciones son cadenas
            # (`from __future__ import annotations`, necesario en Python 3.8); se reconstruye el dataclass.
            config = DiffusionConfig(**config)
        self.config = config
        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config)

        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
       
        # Note: This check is covered in the post-init of the config but have a sanity check just in case.
        if len(image_keys) != 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} only handles one image for now. Got image keys {image_keys}."
            )
        self.input_image_key = image_keys[0]

        self.reset()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.image": deque(maxlen=self.config.n_obs_steps),
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        # Ensemble temporal: descartar chunks del episodio anterior (si no, se promedian con el nuevo episodio)
        if hasattr(self, "active_chunks"):
            self.active_chunks = []

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:

        """n_obs_steps se odnosi na broj koraka koji se uzimaju u obzir prilikom generisanja akcija. Ako
        kes nije popunjenj observacija se kopira dok se sva mjesta ne popune. Kes sluzi kao memorija, kako bi na 
        trenutnu akciju imala uticaj i naredna akcija. 
        Ti koraci se kesiraju i samo n koraka se koristi prilikom generisanja akcija. Generise se onoliko akcija koliko je 
        definisano sa horizonom, ali se ne izvrsavaju sve te akcije vec onoliko akcija koliko je definisano sa n_actions_step
        """


        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... |n-o+1+h|
            |observation is used | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps < horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        # Estado crudo (sin normalizar) del instante actual: ancla de las acciones "relative_ee".
        raw_state = batch["observation.state"]
        batch = self.normalize_inputs(batch)
        batch["observation.image"] = batch[self.input_image_key]

        self._queues = populate_queues(self._queues, batch)

        if getattr(self.config, 'use_temporal_ensemble', False):
            if not hasattr(self, 'active_chunks'):
                self.active_chunks = []
                self.te_k = getattr(self.config, 'temporal_ensemble_k', 0.01)

            # 1. Update ages and remove chunks that no longer cover the current timestep
            for chunk_dict in self.active_chunks:
                chunk_dict['age'] += 1
            self.active_chunks = [c for c in self.active_chunks if c['age'] < c['chunk'].shape[0]]
            
            # 2. Query model if it's time (simulated by checking the queue)
            if len(self._queues["action"]) == 0:
                batch_for_model = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
                actions = self.diffusion.generate_actions(batch_for_model, return_full_horizon=True)
                actions = self.unnormalize_outputs({"action": actions})["action"]
                actions = self.actions_to_absolute(actions, raw_state)
                
                # Transpose to [H_valid, B, action_dim]
                actions_transposed = actions.transpose(0, 1)
                
                self.active_chunks.append({'chunk': actions_transposed, 'age': 0})
                
                # Fill the queue with dummy values to keep the counting mechanism
                dummy_actions = torch.zeros_like(actions_transposed[:self._queues["action"].maxlen])
                self._queues["action"].extend(dummy_actions)

            # 3. Ensemble overlapping predictions
            actions_for_curr_step = []
            weights = []
            for chunk_dict in self.active_chunks:
                age = chunk_dict['age']
                actions_for_curr_step.append(chunk_dict['chunk'][age])
                weights.append(math.exp(-self.te_k * age))
                
            self._queues["action"].popleft()
            
            if not actions_for_curr_step:
                return torch.zeros((1, self.config.output_shapes["action"][0]), device=batch["observation.state"].device)
                
            actions_tensor = torch.stack(actions_for_curr_step, dim=0) # [num_chunks, B, action_dim]
            if actions_tensor.shape[-1] >= 7:
                # q y -q son la misma rotación: alinear signos antes de promediar (si no, se anulan) y renormalizar.
                ref = self._quaternion_reference(actions_tensor)
                if ref is None:
                    ref = actions_tensor[0, :, 3:7]
                actions_tensor = actions_tensor.clone()
                actions_tensor[..., 3:7] = _quat_canonicalize(actions_tensor[..., 3:7], "home", ref)
            weights_tensor = torch.tensor(weights, dtype=actions_tensor.dtype, device=actions_tensor.device)
            weights_tensor = weights_tensor / weights_tensor.sum()
            
            ensembled_action = (actions_tensor * weights_tensor.view(-1, 1, 1)).sum(dim=0) # [B, action_dim]
            if ensembled_action.shape[-1] >= 7:
                ensembled_action[..., 3:7] = _quat_normalize(ensembled_action[..., 3:7])
            return ensembled_action
            
        else:
            if len(self._queues["action"]) == 0:
                # stack n latest observations from the queue
                batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
                actions = self.diffusion.generate_actions(batch)
    
                # TODO(rcadene): make above methods return output dictionary?
                actions = self.unnormalize_outputs({"action": actions})["action"]
                actions = self.actions_to_absolute(actions, raw_state)
    
                # Add only up to maxlen to prevent .extend() from dropping the earliest actions!
                actions_to_queue = actions.transpose(0, 1)[:self._queues["action"].maxlen]
                self._queues["action"].extend(actions_to_queue)
    
            action = self._queues["action"].popleft()
            return action

    # ------------------------------------------------------------------------------------------
    # Representación de la acción y convención de cuaterniones
    # ------------------------------------------------------------------------------------------
    def _quaternion_reference(self, like: Tensor) -> Tensor | None:
        ref = getattr(self.config, "quaternion_reference", None)
        if getattr(self.config, "quaternion_hemisphere", "qw_positive") != "home" or ref is None:
            return None
        return _quat_normalize(torch.as_tensor(ref, dtype=like.dtype, device=like.device))

    def actions_to_absolute(self, actions: Tensor, raw_state: Tensor) -> Tensor:
        """Lleva acciones desnormalizadas ``[B, H, A]`` a la representación absoluta del estado.

        ``raw_state [B, D]`` es el estado SIN normalizar del instante en que se generó el chunk (para
        ``relative_ee``: pose actual del efector, ancla de la composición ``T_actual · T_rel``). Los cuaterniones se
        normalizan y se canonizan con la convención del checkpoint (``quaternion_hemisphere``).
        """
        if actions.shape[-1] < 7:
            return actions
        out = actions.clone()
        q = _quat_normalize(out[..., 3:7])
        if getattr(self.config, "action_representation", "absolute") == "relative_ee":
            p_c = raw_state[:, None, :3].to(out.dtype)
            q_c = _quat_normalize(raw_state[:, None, 3:7].to(out.dtype)).expand_as(q)
            out[..., :3] = p_c + _quat_rotate(q_c, out[..., :3])
            q = _quat_normalize(_quat_multiply(q_c, q))
        ref = self._quaternion_reference(out)
        out[..., 3:7] = _quat_canonicalize(q, getattr(self.config, "quaternion_hemisphere", "qw_positive"), ref)
        return out

    @torch.no_grad()
    def enable_cuda_graph(self, batch_size: int = 1) -> bool:
        """Inferencia con CUDA Graphs (ver ``DiffusionModel.enable_cuda_graph``). Llamar después de ``.to(cuda)``."""
        return self.diffusion.enable_cuda_graph(batch_size)

    def predict_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predicción offline (evaluación): ``batch`` con historial ``[B, n_obs, ...]`` SIN normalizar.

        Devuelve el horizonte completo ejecutable ``[B, horizon - n_obs + 1, A]`` en representación absoluta, igual
        que lo recibiría ``policy_node`` (sin colas ni ensemble temporal).
        """
        raw_state = batch["observation.state"][:, -1]
        nb = self.normalize_inputs(dict(batch))
        nb["observation.image"] = nb[self.input_image_key]
        actions = self.diffusion.generate_actions(
            {"observation.image": nb["observation.image"], "observation.state": nb["observation.state"]},
            return_full_horizon=True)
        actions = self.unnormalize_outputs({"action": actions})["action"]
        return self.actions_to_absolute(actions, raw_state)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        
        batch = self.normalize_inputs(batch)
        batch["observation.image"] = batch[self.input_image_key]
            
        batch = self.normalize_targets(batch)
        # print('action: ', batch['action'])
        loss = self.diffusion.compute_loss(batch)
        # print(loss)
        return {"loss": loss}


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")



class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        self.rgb_encoder = DiffusionRgbEncoder(config)
        state_dim = config.input_shapes["observation.state"][0]
        self.unet = DiffusionConditionalUnet1d(
            config,
            global_cond_dim=(self.rgb_encoder.feature_dim + state_dim) * config.n_obs_steps,
        )

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps
        self._cuda_graph = None
        self._cuda_graph_enabled = False
        # El candado se crea al activar el grafo: un threading.Lock como atributo impediría copy.deepcopy (EMA)
        self._cuda_graph_lock = None

    # ========= CUDA Graphs (inferencia en el robot) ============
    def enable_cuda_graph(self, batch_size: int = 1) -> bool:
        """Captura en un grafo CUDA el encoder de imagen y el bucle DDIM completo (batch fijo, sin gradiente).

        En la Jetson Orin Nano la inferencia está limitada por el costo de lanzar cientos de kernels pequeños desde la
        CPU: una llamada a la U-Net tarda 18 ms normal y 9 ms en grafo. Reproducir el grafo da el MISMO resultado que
        la ejecución normal (diferencia 0 medida en la Orin) en la mitad de tiempo. El ruido inicial se genera fuera
        del grafo en cada llamada, con el mismo generador que el modo normal, así que el muestreo sigue siendo
        estocástico e idéntico para una misma semilla.

        Solo para inferencia en CUDA, con el modelo ya en el dispositivo y en ``eval()``. Si cambian
        ``num_inference_steps`` o el tamaño de batch, ``generate_actions`` vuelve a capturar. Devuelve False sin
        cambiar nada si el modelo no está en CUDA.
        """
        if get_device_from_parameters(self).type != "cuda":
            self.disable_cuda_graph()
            return False
        if self._cuda_graph_lock is None:
            self._cuda_graph_lock = threading.Lock()
        self._cuda_graph_enabled = True
        with self._cuda_graph_lock:
            self._capture_cuda_graph(int(batch_size))
        return True

    def disable_cuda_graph(self) -> None:
        self._cuda_graph_enabled = False
        self._cuda_graph = None

    def __getstate__(self):
        # Ni el grafo CUDA ni su candado se copian (copy.deepcopy del EMA, pickle): la copia queda en modo normal.
        state = self.__dict__.copy()
        state["_cuda_graph"] = None
        state["_cuda_graph_enabled"] = False
        state["_cuda_graph_lock"] = None
        return state

    def _capture_cuda_graph(self, batch_size: int) -> None:
        cfg = self.config
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        image_key = [k for k in cfg.input_shapes if k.startswith("observation.image")][0]
        b, n = batch_size, cfg.n_obs_steps
        static = {
            "image": torch.zeros((b * n, *cfg.input_shapes[image_key]), dtype=dtype, device=device),
            "state": torch.zeros((b, n, cfg.input_shapes["observation.state"][0]), dtype=dtype, device=device),
            "noise": torch.zeros((b, cfg.horizon, cfg.output_shapes["action"][0]), dtype=dtype, device=device),
        }
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = [int(t) for t in self.noise_scheduler.timesteps]

        def run() -> Tensor:
            # Mismo cálculo que generate_actions + conditional_sample (sin la generación del ruido)
            feats = self.rgb_encoder(static["image"]).reshape(b, -1)
            global_cond = torch.cat([feats, static["state"].flatten(start_dim=1)], dim=-1)
            x = static["noise"]
            for t in timesteps:
                out = self.unet(x, torch.full((b,), t, dtype=torch.long, device=device), global_cond=global_cond)
                x = self.noise_scheduler.step(out, t, x).prev_sample
            return x

        with torch.no_grad():
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):          # calentamiento fuera del stream por defecto (requisito de captura)
                for _ in range(3):
                    run()
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run()
        self._cuda_graph = {"graph": graph, "static": static, "out": out, "steps": self.num_inference_steps,
                            "batch": b, "device": device}

    def _sample_with_cuda_graph(self, batch: dict[str, Tensor]) -> Tensor | None:
        """Muestra con el grafo; None si no aplica (desactivado, en entrenamiento, otro dispositivo)."""
        if not self._cuda_graph_enabled or self.training:
            return None
        state = batch["observation.state"]
        if state.device.type != "cuda":
            return None
        with self._cuda_graph_lock:
            g = self._cuda_graph
            if g is None or g["steps"] != self.num_inference_steps or g["batch"] != state.shape[0] \
                    or g["device"] != state.device:
                self._capture_cuda_graph(state.shape[0])
                g = self._cuda_graph
            st = g["static"]
            st["image"].copy_(einops.rearrange(batch["observation.image"], "b n ... -> (b n) ..."))
            st["state"].copy_(state)
            st["noise"].copy_(torch.randn(st["noise"].shape, dtype=st["noise"].dtype, device=st["noise"].device))
            g["graph"].replay()
            return g["out"].clone()

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.output_shapes["action"][0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def generate_actions(self, batch: dict[str, Tensor], return_full_horizon=False) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.image": (B, n_obs_steps, C, H, W)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Con enable_cuda_graph(): encoder + bucle DDIM en un grafo CUDA (mismo resultado, ~2x más rápido en la Orin)
        sample = self._sample_with_cuda_graph(batch)
        if sample is None:
            # Extract image feature (first combine batch and sequence dims).
            img_features = self.rgb_encoder(einops.rearrange(batch["observation.image"], "b n ... -> (b n) ..."))

            # Separate batch and sequence dims.
            img_features = einops.rearrange(img_features, "(b n) ... -> b n ...", b=batch_size)

            # Ahora sí usamos el state (Pose Cartesiana Real) también en inferencia
            state_features = batch["observation.state"].flatten(start_dim=1)
            global_cond = torch.cat([img_features.flatten(start_dim=1), state_features], dim=-1)

            # run sampling
            sample = self.conditional_sample(batch_size, global_cond=global_cond)

        # `horizon` steps worth of actions (from the first observation).
        actions = sample[..., : self.config.output_shapes["action"][0]]
        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        
        if return_full_horizon:
            actions = actions[:, start:]
        else:
            end = start + self.config.n_action_steps
            actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.image": (B, n_obs_steps, C, H, W)
            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "observation.image", "action", "action_is_pad"})

        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps 

        img_features = self.rgb_encoder(einops.rearrange(batch["observation.image"], "b n ... -> (b n) ..."))
        # print('____________ After RGB encoder:  ____________', img_features)
        # breakpoint()
        # Separate batch and sequence dims.
        img_features = einops.rearrange(img_features, "(b n) ... -> b n ...", b=batch_size)
        
        # Ahora sí usamos el state (Pose Cartesiana Real)
        state_features = batch["observation.state"].flatten(start_dim=1)
        global_cond = torch.cat([img_features.flatten(start_dim=1), state_features], dim=-1)

        trajectory = batch["action"]

        # Forward diffusion.
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        # print(noisy_trajectory)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        # loss = F.l1_loss(pred, target, reduction="none")
        # print('LOSS: ', pred, target)
        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    f"You need to provide 'action_is_pad' in the batch when {self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encoder an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

       
        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        assert len(image_keys) == 1
        image_key = image_keys[0]
        dummy_input_h_w = (
            config.crop_shape if config.crop_shape is not None else config.input_shapes[image_key][1:]
        )
        dummy_input = torch.zeros(size=(1, config.input_shapes[image_key][0], *dummy_input_h_w))
        with torch.inference_mode():
            dummy_feature_map = self.backbone(dummy_input)

        feature_map_shape = tuple(dummy_feature_map.shape[1:])
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.output_shapes["action"][0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:])  # strict=True requiere Python>=3.10; ambas listas tienen igual longitud
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.output_shapes["action"][0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out




