import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('visuomotor_bringup')
    
    gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='false',
        description='Lanzar visor gráfico rtabmap_viz'
    )
    
    camera_config = os.path.join(pkg_share, 'config', 'realsense', 'd415_camera_params.yaml')
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

    # 3. Visor opcional rtabmap_viz
    viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        output='screen',
        parameters=[{
            'frame_id': 'camera_link',
            'odom_frame_id': 'odom',
            'approx_sync': False,
            'queue_size': 10
        }],
        remappings=[
            ('rgb/image', '/camera/camera/color/image_raw'),
            ('depth/image', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('rgb/camera_info', '/camera/camera/color/camera_info'),
            ('odom', '/rtabmap/odom')
        ],
        condition=IfCondition(LaunchConfiguration('gui'))
    )

    return LaunchDescription([
        gui_arg,
        realsense_node,
        odom_node,
        viz_node
    ])
