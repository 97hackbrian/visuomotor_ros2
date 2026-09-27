# ObservationData.msg
# Estructura que sincroniza modalidades sensoriales para la política.

std_msgs/Header header
sensor_msgs/Image[] images          # Múltiples cámaras si es necesario
sensor_msgs/JointState joint_state  # Propiocepción articular
geometry_msgs/PoseStamped end_effector_pose # Pose cartesiana del efector
