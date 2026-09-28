from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray

class ActionExecutor:
    def __init__(self, node):
        self.node = node
        # Se publica la acción en target_frame para que Isaac Sim / IK la ejecute
        self.pose_pub = node.create_publisher(PoseStamped, 'target_frame', 10)
        # Se publica el estado del gripper para que Isaac Sim lo abra/cierre
        self.gripper_pub = node.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        
    def execute(self, action_chunk):
        """
        action_chunk: visuomotor_msgs.msg.ActionChunk
        """
        if not action_chunk.poses:
            return
            
        # Example of taking the first action in receding horizon
        target = PoseStamped()
        target.header = action_chunk.header
        target.pose = action_chunk.poses[0]
        self.pose_pub.publish(target)
        
        # Publicar estado del gripper si existe
        if action_chunk.gripper_states:
            gripper_val = action_chunk.gripper_states[0]
            gripper_msg = Float64MultiArray()
            # Asumimos que el array contiene dos valores o los que requiera el controlador
            gripper_msg.data = [float(gripper_val)]
            self.gripper_pub.publish(gripper_msg)
