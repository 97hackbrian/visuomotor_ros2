cmake_minimum_required(VERSION 3.8)
project(visuomotor_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Módulo Python puro con la lógica del xArm5 (visuomotor_bringup.xarm5_interface), testeable sin ROS
ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY
  launch
  rviz
  urdf
  config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/sixd_speed_limiter
  scripts/pose_smoother.py
  scripts/xarm5_bridge.py
  scripts/xarm5_teleop_marker.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
