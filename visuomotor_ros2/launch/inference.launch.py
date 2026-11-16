import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('visuomotor_ros2')
    default_config = os.path.join(pkg_share, 'config', 'policy_params.yaml')

    ckpt_path_val = LaunchConfiguration('checkpoint_path').perform(context)

    node_params = [default_config]
    if ckpt_path_val.strip():
        node_params.append({'checkpoint_path': ckpt_path_val})

    policy_node = Node(
        package='visuomotor_ros2',
        executable='policy_node',
        name='policy_node',
        output='screen',
        parameters=node_params
    )

    return [policy_node]


def generate_launch_description():
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint_path',
        default_value='',
        description='Ruta absoluta al checkpoint del modelo a inferir (si se omite, se lee de policy_params.yaml)'
    )

    return LaunchDescription([
        checkpoint_arg,
        OpaqueFunction(function=launch_setup)
    ])
