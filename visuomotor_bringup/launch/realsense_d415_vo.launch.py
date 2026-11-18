import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('visuomotor_bringup')
    
    use_ultra = context.perform_substitution(LaunchConfiguration('ultra_precision')).strip().lower() in ('true', '1')
    gui = context.perform_substitution(LaunchConfiguration('gui')).strip().lower() in ('true', '1')
    
    camera_config = os.path.join(pkg_share, 'config', 'realsense', 'd415_camera_params.yaml')
    if use_ultra:
        vo_config = os.path.join(pkg_share, 'config', 'realsense', 'rtabmap_vo_ultra_precision.yaml')
    else:
        vo_config = os.path.join(pkg_share, 'config', 'realsense', 'rtabmap_vo_params.yaml')

    # 1. Driver RealSense D415
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        namespace='camera',
        parameters=[camera_config],
        output='screen'
    )

    # 2. RTAB-Map RGB-D Odometry (Headless)
    odom_node = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='rgbd_odometry',
        output='screen',
        parameters=[vo_config],
        remappings=[
            ('rgb/image', '/camera/camera/color/image_raw'),
            ('depth/image', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('rgb/camera_info', '/camera/camera/color/camera_info'),
            ('odom', '/rtabmap/odom')
        ]
    )

    nodes = [realsense_node, odom_node]

    # 3. Visor gráfico RViz2 con configuración vo_vis.rviz
    if gui:
        rviz_config = os.path.join(pkg_share, 'rviz', 'vo_vis.rviz')
        ws_rviz = '/ros/.rviz2/vo_vis.rviz'
        if os.path.exists(ws_rviz):
            rviz_config = ws_rviz

        rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config]
        )
        nodes.append(rviz_node)

    return nodes

def generate_launch_description():
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='false',
        description='Lanzar visor gráfico RViz2 con vo_vis.rviz'
    )
    
    ultra_precision_arg = DeclareLaunchArgument(
        'ultra_precision',
        default_value='false',
        description='Usar configuración de ultra-precisión (SIFT + Subpíxel + g2o BA)'
    )

    return LaunchDescription([
        gui_arg,
        ultra_precision_arg,
        OpaqueFunction(function=launch_setup)
    ])
