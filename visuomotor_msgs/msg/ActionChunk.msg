# ActionChunk.msg
# Representa un bloque de acciones futuras predecidas por la política.
# Cada pose de la matriz corresponde a un paso futuro en el horizonte temporal.

std_msgs/Header header
geometry_msgs/Pose[] poses
float32[] gripper_states # Estado del gripper si aplica (ej. 0.0 cerrado, 1.0 abierto)
