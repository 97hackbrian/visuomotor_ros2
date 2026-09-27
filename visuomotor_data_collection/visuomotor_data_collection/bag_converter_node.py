import sys
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from visuomotor_data_collection.utils.zarr_storage import ZarrStorage

def convert_bag(bag_uri, zarr_output_path, img_topic, action_topic):
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    storage = ZarrStorage(zarr_output_path)
    buffer = []
    
    topics_and_types = reader.get_all_topics_and_types()
    type_dict = {t.name: get_message(t.type) for t in topics_and_types}

    while reader.has_next():
        (topic, data, stamp) = reader.read_next()
        msg_type = type_dict.get(topic)
        if msg_type is None:
            continue
        msg = deserialize_message(data, msg_type)
        # TODO: Implementar lógica de sincronización temporal e inserción en el buffer
        pass
    
    storage.append_episode(buffer)
    print(f"Extracción finalizada. Archivo {bag_uri} procesado hacia {zarr_output_path}")

def main():
    if len(sys.argv) < 3:
        print("Sintaxis requerida: ros2 run visuomotor_data_collection bag_converter <ruta_bag> <ruta_zarr>")
        return
    convert_bag(sys.argv[1], sys.argv[2], '/camera/image_raw', '/action_cmd')

if __name__ == '__main__':
    main()
