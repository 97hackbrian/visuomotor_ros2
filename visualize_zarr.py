import zarr
import cv2
import sys
import os

def create_video(zarr_path, output_video, episode_id=None):
    root = zarr.open_group(zarr_path, mode='r')
    episodes = sorted(list(root.groups()))
    
    if not episodes:
        print("No hay episodios en el dataset.")
        return
        
    # Obtener el último episodio si no se especificó uno
    if episode_id is None:
        episode_name = episodes[-1][0] # episodes es una lista de tuplas (nombre, grupo)
    else:
        episode_name = f"episode_{episode_id}"
        
    if episode_name not in root:
        print(f"El episodio {episode_name} no existe.")
        return
        
    images = root[episode_name]['image'][:]
    
    print(f"Exportando {episode_name} ({len(images)} frames) a {output_video}...")
    
    height, width, layers = images[0].shape
    # mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))
    
    for img in images:
        # Convertir RGB (guardado en el zarr) a BGR (que usa cv2 para guardar)
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(bgr_img)
        
    video.release()
    print("¡Video guardado exitosamente!")

if __name__ == "__main__":
    zarr_dir = "demonstrations.zarr"
    out_file = "ultimo_episodio.mp4"
    
    if len(sys.argv) > 1:
        out_file = f"episodio_{sys.argv[1]}.mp4"
        create_video(zarr_dir, out_file, episode_id=sys.argv[1])
    else:
        create_video(zarr_dir, out_file)
