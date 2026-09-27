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
    x = actions[:, 0]
    y = actions[:, 1]
    z = actions[:, 2]
    gripper = actions[:, 7]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plotear toda la trayectoria en una línea gris tenue
    ax.plot(x, y, z, color='gray', alpha=0.5, label='Trayectoria')
    
    # Encontrar índices donde el gripper cambia de estado
    gripper_closed = gripper < -0.005 # umbral para -0.01
    
    # Puntos normales (gripper abierto)
    open_idx = ~gripper_closed
    ax.scatter(x[open_idx], y[open_idx], z[open_idx], color='blue', s=10, alpha=0.3, label='Gripper Abierto (0.0)')
    
    # Puntos de agarre (gripper cerrado)
    close_idx = gripper_closed
    if np.any(close_idx):
        ax.scatter(x[close_idx], y[close_idx], z[close_idx], color='red', s=20, alpha=0.8, label='Gripper Cerrado (-0.01)')
        
        # Encontrar el punto exacto de activación (cierre)
        diff = np.diff(gripper_closed.astype(int))
        activations = np.where(diff == 1)[0] + 1
        deactivations = np.where(diff == -1)[0] + 1
        
        if len(activations) > 0:
            ax.scatter(x[activations], y[activations], z[activations], color='magenta', s=100, marker='*', edgecolor='black', label='Activación (Cierra)')
        if len(deactivations) > 0:
            ax.scatter(x[deactivations], y[deactivations], z[deactivations], color='cyan', s=100, marker='X', edgecolor='black', label='Desactivación (Abre)')

    # Punto de Inicio
    ax.scatter(x[0], y[0], z[0], color='green', s=150, marker='^', edgecolor='black', label='Inicio')
    
    # Punto de Fin
    ax.scatter(x[-1], y[-1], z[-1], color='orange', s=150, marker='v', edgecolor='black', label='Fin')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Forzar aspect ratio 1:1:1 para que no se deforme (evita el "efecto L")
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
    
    ax.set_title(f'Trayectoria 3D del Efector Final - {episode_name}')
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
