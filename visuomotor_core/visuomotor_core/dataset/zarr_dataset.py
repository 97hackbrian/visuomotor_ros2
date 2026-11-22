import torchvision.transforms as transforms
import zarr
import torch
from torch.utils.data import Dataset
import numpy as np

from visuomotor_core.utils.rotations import relative_pose


def episode_sort_key(name: str):
    parts = name.split('_')
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return name


def list_episodes(root) -> list:
    """Nombres de episodio de un dataset Zarr ordenados numéricamente (``episode_0``, ``episode_1``, ...)."""
    group = zarr.open_group(root, mode='r') if isinstance(root, str) else root
    return sorted(group.group_keys(), key=episode_sort_key)


def split_episodes(names: list, val_ratio: float, seed: int = 0):
    """Partición train/val ESTRICTA por episodio completo (nunca por frames: fuga temporal).

    Selección estratificada: se reparten los episodios de validación a lo largo del orden de grabación (sesiones
    distintas) con un desfase aleatorio reproducible.
    """
    names = sorted(names, key=episode_sort_key)
    n_val = int(round(len(names) * float(val_ratio)))
    if n_val <= 0:
        return names, []
    if n_val >= len(names):
        raise ValueError(f"val_ratio={val_ratio} deja 0 episodios de entrenamiento ({len(names)} episodios en total)")
    rng = np.random.default_rng(seed)
    stride = len(names) / n_val
    offset = rng.uniform(0, stride)
    val_idx = sorted({min(len(names) - 1, int(offset + i * stride)) for i in range(n_val)})
    val = [names[i] for i in val_idx]
    train = [n for n in names if n not in set(val)]
    return train, val


class ZarrDataset(Dataset):
    """Dataset de ventanas (historial de observación + horizonte de acción) sobre episodios Zarr.

    Parámetros nuevos (valores por defecto = comportamiento histórico):
        episodes: subconjunto explícito de episodios (para particiones train/val por episodio).
        action_representation: "absolute" o "relative_ee" (acción relativa a la pose actual, estilo UMI).
        stats: estadísticas precalculadas (p. ej. las de train para el conjunto de validación).
        image_size, proprio_dropout_prob: antes fijos en el código.
        color_jitter: dict de ``torchvision.transforms.ColorJitter`` (solo entrenamiento) o None.
        std_floor: rango (max - min) bajo el cual una dimensión del estado se considera CONSTANTE (p. ej. gripper
            pasivo en 0.0) y se normaliza con std = 1, para que una lectura de -1e-4 en el robot no dé z = -99. Las
            dimensiones con variación pequeña pero real conservan su std (comportamiento histórico).
    """

    def __init__(self, root, n_obs_steps=2, h_act=16, max_episodes=0, episodes=None,
                 action_representation="absolute", stats=None, image_size=256, proprio_dropout_prob=0.15,
                 color_jitter=None, std_floor=1e-6):
        self.root_path = root
        self.root = zarr.open_group(root, mode='r')

        names = list_episodes(self.root)
        if episodes is not None:
            missing = set(episodes) - set(names)
            if missing:
                raise ValueError(f"Episodios inexistentes en {root}: {sorted(missing)[:5]}")
            names = sorted(episodes, key=episode_sort_key)
        if max_episodes > 0:
            names = names[:max_episodes]
        self.episodes = [(n, self.root[n]) for n in names]

        if action_representation not in ("absolute", "relative_ee"):
            raise ValueError(f"action_representation desconocida: {action_representation!r}")
        self.action_representation = action_representation
        self.n_obs_steps = n_obs_steps
        self.h_act = h_act
        self.image_size = int(image_size)
        self.proprio_dropout_prob = float(proprio_dropout_prob)
        self.std_floor = float(std_floor)
        self.color_jitter = transforms.ColorJitter(**color_jitter) if color_jitter else None
        self._resize = transforms.Resize((self.image_size, self.image_size), antialias=True)

        self.frames = []
        for ep_name, ep_group in self.episodes:
            num_frames = ep_group['image'].shape[0]
            # Permitir indexar hasta el final del episodio mediante copy-padding
            for i in range(n_obs_steps - 1, num_frames):
                self.frames.append((ep_name, i))

        self.stats = stats if stats is not None else self._compute_stats()

    # ------------------------------------------------------------------------------------------
    def window_indices(self, num_frames: int, frame_idx):
        """Índices del horizonte de acción con copy-padding (repite el último frame del episodio)."""
        frame_idx = np.asarray(frame_idx)
        k = np.arange(self.h_act)
        return np.minimum(frame_idx[..., None] + k, num_frames - 1)

    def to_representation(self, current_pose: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Acciones absolutas ``[..., H, A]`` -> representación configurada (ancla: ``current_pose [..., >=7]``)."""
        if self.action_representation == "absolute" or actions.shape[-1] < 7:
            return actions
        out = np.array(actions, dtype=np.float64, copy=True)
        cur = np.asarray(current_pose, dtype=np.float64)[..., None, :]
        dp, dq = relative_pose(cur[..., :3], cur[..., 3:7], out[..., :3], out[..., 3:7])
        out[..., :3] = dp
        out[..., 3:7] = dq
        return out.astype(np.float32)

    def _compute_stats(self):
        all_states = []
        all_actions = []
        for ep_name, ep_group in self.episodes:
            states = ep_group['state'][:]
            actions = ep_group['action'][:]
            all_states.append(states)
            if self.action_representation == "absolute":
                all_actions.append(actions)
            else:
                idx = np.arange(self.n_obs_steps - 1, len(actions))
                win = actions[self.window_indices(len(actions), idx)]            # [N, H, A]
                all_actions.append(self.to_representation(states[idx], win).reshape(-1, actions.shape[-1]))

        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)

        constant = (all_states.max(axis=0) - all_states.min(axis=0)) <= self.std_floor
        state_std = np.where(constant, 1.0, all_states.std(axis=0) + 1e-6)
        self.constant_state_dims = [int(i) for i in np.flatnonzero(constant)]

        stats = {
            "observation.image": {
                "mean": torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1),
                "std": torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1),
            },
            "observation.state": {
                "min": torch.tensor(all_states.min(axis=0), dtype=torch.float32),
                "max": torch.tensor(all_states.max(axis=0), dtype=torch.float32),
                "mean": torch.tensor(all_states.mean(axis=0), dtype=torch.float32),
                "std": torch.tensor(state_std, dtype=torch.float32),
            },
            "action": {
                "min": torch.tensor(all_actions.min(axis=0), dtype=torch.float32),
                "max": torch.tensor(all_actions.max(axis=0), dtype=torch.float32),
                "mean": torch.tensor(all_actions.mean(axis=0), dtype=torch.float32),
                "std": torch.tensor(all_actions.std(axis=0) + 1e-6, dtype=torch.float32),
            }
        }
        return stats

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        ep_name, frame_idx = self.frames[idx]
        group = self.root[ep_name]

        imgs = group['image'][frame_idx - self.n_obs_steps + 1 : frame_idx + 1]
        imgs = torch.tensor(imgs, dtype=torch.float32) / 255.0
        imgs = imgs.permute(0, 3, 1, 2)
        if imgs.shape[-2:] != (self.image_size, self.image_size):
            imgs = self._resize(imgs)
        if self.color_jitter is not None:
            # mismos parámetros para todo el historial (una llamada = un muestreo de parámetros)
            imgs = self.color_jitter(imgs).clamp(0.0, 1.0)

        states_np = group['state'][frame_idx - self.n_obs_steps + 1 : frame_idx + 1]
        states = torch.tensor(states_np, dtype=torch.float32)

        # Proprioception Dropout
        if self.proprio_dropout_prob > 0 and torch.rand(1).item() < self.proprio_dropout_prob:
            states = torch.zeros_like(states)

        total_ep_frames = group['action'].shape[0]
        actual_actions = group['action'][frame_idx : min(total_ep_frames, frame_idx + self.h_act)]
        actual_len = len(actual_actions)

        if actual_len < self.h_act:
            pad_len = self.h_act - actual_len
            pad_actions = np.repeat(actual_actions[-1:], pad_len, axis=0)
            actions = np.concatenate([actual_actions, pad_actions], axis=0)
            action_is_pad = torch.zeros(self.h_act, dtype=torch.bool)
            action_is_pad[actual_len:] = True
        else:
            actions = actual_actions
            action_is_pad = torch.zeros(self.h_act, dtype=torch.bool)

        # el ancla es la pose REAL actual (antes del dropout propioceptivo)
        actions = self.to_representation(states_np[-1], actions)
        actions = torch.tensor(actions, dtype=torch.float32)

        return {
            "observation.image": imgs,
            "observation.state": states,
            "action": actions,
            "action_is_pad": action_is_pad
        }
