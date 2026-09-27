import message_filters
from sensor_msgs.msg import Image, JointState

class ObservationBuffer:
    def __init__(self, node, h_obs):
        self.node = node
        self.h_obs = h_obs
        
        self.image_sub = message_filters.Subscriber(node, Image, 'camera/image_raw')
        self.joint_sub = message_filters.Subscriber(node, JointState, 'joint_states')
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.joint_sub], 10, 0.1)
        self.ts.registerCallback(self._sync_cb)
        
        self.buffer = []
        
    def _sync_cb(self, img_msg, joint_msg):
        self.buffer.append({'image': img_msg, 'joint': joint_msg})
        if len(self.buffer) > self.h_obs:
            self.buffer.pop(0)
            
    def get_recent_observations(self):
        if len(self.buffer) < self.h_obs:
            return None
        return self.buffer.copy()
