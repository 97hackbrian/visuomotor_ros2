import zarr
import numpy as np
import os

class ZarrStorage:
    def __init__(self, path):
        self.path = path
        # Usaremos zarr.open_group para inicializar o abrir el grupo Zarr
        self.root = zarr.open_group(self.path, mode='a')
        
    def append_episode(self, buffer):
        """
        buffer: List of dicts with 'image', 'state', 'action'
        """
        if not buffer:
            return
            
        episode_idx = len(list(self.root.groups()))
        episode_group = self.root.create_group(f"episode_{episode_idx}")
        
        images = np.stack([step['image'] for step in buffer])
        states = np.stack([step['state'] for step in buffer])
        actions = np.stack([step['action'] for step in buffer])
        
        episode_group.create_dataset('image', data=images, chunks=(1, *images.shape[1:]), dtype='uint8')
        episode_group.create_dataset('state', data=states, chunks=(1, *states.shape[1:]), dtype='float32')
        episode_group.create_dataset('action', data=actions, chunks=(1, *actions.shape[1:]), dtype='float32')
