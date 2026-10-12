import torchvision.transforms as transforms
import zarr
import torch
from torch.utils.data import Dataset
import numpy as np

class ZarrDataset(Dataset):
    def __init__(self, root, n_obs_steps=2, h_act=16):
        self.root_path = root
        self.root = zarr.open_group(root, mode='r')
        self.episodes = sorted(list(self.root.groups()))
        
        self.n_obs_steps = n_obs_steps
        self.h_act = h_act
        
        self.frames = []
        for ep_name, ep_group in self.episodes:
            num_frames = ep_group['image'].shape[0]
            # Permitir indexar hasta el final del episodio mediante copy-padding
            for i in range(n_obs_steps - 1, num_frames):
                self.frames.append((ep_name, i))
                
        self.stats = self._compute_stats()

    def _compute_stats(self):
        all_states = []
        all_actions = []
        for ep_name, ep_group in self.episodes:
            all_states.append(ep_group['state'][:])
            all_actions.append(ep_group['action'][:])
            
        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        stats = {
            "observation.image": {
                "mean": torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1),
                "std": torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1),
            },
            "observation.state": {
                "min": torch.tensor(all_states.min(axis=0), dtype=torch.float32),
                "max": torch.tensor(all_states.max(axis=0), dtype=torch.float32),
                "mean": torch.tensor(all_states.mean(axis=0), dtype=torch.float32),
                "std": torch.tensor(all_states.std(axis=0) + 1e-6, dtype=torch.float32),
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
        # 🚨 FIX: La cámara es 720p (1280x720). Resize a 256x256 ANTES del recorte (crop)
        imgs = transforms.Resize((256, 256), antialias=True)(imgs)
        
        states = group['state'][frame_idx - self.n_obs_steps + 1 : frame_idx + 1]
        states = torch.tensor(states, dtype=torch.float32)
        
        # Proprioception Dropout (15% probability)
        if torch.rand(1).item() < 0.15:
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
            
        actions = torch.tensor(actions, dtype=torch.float32)
        
        return {
            "observation.image": imgs,
            "observation.state": states,
            "action": actions,
            "action_is_pad": action_is_pad
        }
