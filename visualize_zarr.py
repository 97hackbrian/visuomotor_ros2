import zarr
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_episode(zarr_path, episode_id=None):
    root = zarr.open_group(zarr_path, mode='r')
    episodes = sorted(list(root.groups()))
    
    if not episodes:
        print("No hay episodios en el dataset.")
        return
        
    if episode_id is None:
        episode_name = episodes[-1][0]
    else:
        episode_name = f"episode_{episode_id}"
        
    if episode_name not in root:
        print(f"El episodio {episode_name} no existe.")
        return
        
    images = root[episode_name]['image'][:]
    actions = root[episode_name]['action'][:]  # [x, y, z, qx, qy, qz, qw, gripper]
    states = root[episode_name]['state'][:]    # [x, y, z, qx, qy, qz, qw, gripper]
    
    output_video = f"{episode_name}.mp4"
    print(f"Exportando {episode_name} ({len(images)} frames)...")
    
    # 1. Crear el Video
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))
    
    for img in images:
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(bgr_img)
        
    video.release()
    print(f"¡Video guardado en {output_video}!")
    
    # 2. Generar el Plot 3D
    ax_x = actions[:, 0]
    ax_y = actions[:, 1]
    ax_z = actions[:, 2]
    
    # State: Real physical pose
    if states.shape[1] >= 3:
        sx_x = states[:, 0]
        sx_y = states[:, 1]
        sx_z = states[:, 2]
    
    gripper = actions[:, 7]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotear Action (PID Target)
    ax.plot(ax_x, ax_y, ax_z, color='gray', alpha=0.5, linewidth=2, label='PID Target (Action)')
    
    # Plotear State (Real Physical Pose)
    if states.shape[1] >= 3:
        ax.plot(sx_x, sx_y, sx_z, color='green', alpha=0.8, linestyle='dashed', linewidth=2, label='Real EE Pose (State)')
    
    # Encontrar índices donde el gripper cambia de estado
    gripper_closed = gripper < -0.005 # umbral para -0.01
    
    # Puntos normales (gripper abierto)
    open_idx = ~gripper_closed
    ax.scatter(ax_x[open_idx], ax_y[open_idx], ax_z[open_idx], color='blue', s=10, alpha=0.3, label='Gripper Abierto (0.0)')
    
    # Puntos de agarre (gripper cerrado)
    close_idx = gripper_closed
    if np.any(close_idx):
        ax.scatter(ax_x[close_idx], ax_y[close_idx], ax_z[close_idx], color='red', s=20, alpha=0.8, label='Gripper Cerrado (-0.01)')
        
        diff = np.diff(gripper_closed.astype(int))
        activations = np.where(diff == 1)[0] + 1
        deactivations = np.where(diff == -1)[0] + 1
        
        if len(activations) > 0:
            ax.scatter(ax_x[activations], ax_y[activations], ax_z[activations], color='magenta', s=100, marker='*', edgecolor='black', label='Activación (Cierra)')
        if len(deactivations) > 0:
            ax.scatter(ax_x[deactivations], ax_y[deactivations], ax_z[deactivations], color='cyan', s=100, marker='X', edgecolor='black', label='Desactivación (Abre)')

    ax.scatter(ax_x[0], ax_y[0], ax_z[0], color='green', s=150, marker='^', edgecolor='black', label='Inicio')
    ax.scatter(ax_x[-1], ax_y[-1], ax_z[-1], color='orange', s=150, marker='v', edgecolor='black', label='Fin')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Calcular límites globales combinando actions y states
    all_x = np.concatenate([ax_x, sx_x]) if states.shape[1] >= 3 else ax_x
    all_y = np.concatenate([ax_y, sx_y]) if states.shape[1] >= 3 else ax_y
    all_z = np.concatenate([ax_z, sx_z]) if states.shape[1] >= 3 else ax_z
    
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max()
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
    
    ax.set_title(f'Trayectoria 3D - {episode_name}')
    ax.legend()
    
    plot_file = f"{episode_name}_trajectory.png"
    plt.savefig(plot_file)
    print(f"¡Plot 3D guardado en {plot_file}!")

if __name__ == "__main__":
    zarr_dir = "demonstrations.zarr"
    if len(sys.argv) > 1:
        visualize_episode(zarr_dir, episode_id=sys.argv[1])
    else:
        visualize_episode(zarr_dir)
