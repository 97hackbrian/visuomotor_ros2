from geometry_msgs.msg import PoseStamped

class ActionExecutor:
    def __init__(self, node):
        self.node = node
        self.pose_pub = node.create_publisher(PoseStamped, 'target_pid_pose', 10)
        
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
        
        # Opcional: Publicar estado del gripper si existe
        if action_chunk.gripper_states:
            gripper_val = action_chunk.gripper_states[0]
            # TODO: Enviar gripper_val al controlador del gripper
