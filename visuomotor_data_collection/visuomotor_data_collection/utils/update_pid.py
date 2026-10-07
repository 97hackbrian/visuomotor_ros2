import numpy as np

def compute_orientation_new(self, current_quat, target_quat):
    if not hasattr(self, 'virtual_quat') or self.virtual_quat is None or target_quat is None:
        self.virtual_quat = np.array(current_quat)
        
    if target_quat is None:
        return current_quat
        
    max_rot_step = self.max_speed_rad_s / self.sampling_rate_hz
    
    q1 = self.virtual_quat
    q2 = np.array(target_quat)
    
    dot = np.sum(q1 * q2)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
        
    if dot > 0.9995:
        res = q1 + self.kp_rot * (q2 - q1)
        res = res / np.linalg.norm(res)
        self.virtual_quat = res
        return res
        
    theta_0 = np.arccos(dot)
    
    # Proportional control on angle
    step_theta = theta_0 * self.kp_rot
    if step_theta > max_rot_step:
        t = max_rot_step / theta_0
    else:
        t = self.kp_rot
        
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    res = (s0 * q1) + (s1 * q2)
    res = res / np.linalg.norm(res)
    self.virtual_quat = res
    return res
