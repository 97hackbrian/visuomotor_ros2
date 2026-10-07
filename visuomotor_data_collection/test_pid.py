import numpy as np

def compute_orientation(current_quat, target_quat, kp_rot=1.5):
    q1 = current_quat
    q2 = target_quat
    dot = np.sum(q1 * q2)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        res = q1 + kp_rot * (q2 - q1)
        return res / np.linalg.norm(res)
    
    theta_0 = np.arccos(dot)
    t = kp_rot
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    res = (s0 * q1) + (s1 * q2)
    return res / np.linalg.norm(res)

q_curr = np.array([0.0, 0.0, 0.0, 1.0])
q_targ = np.array([0.0, 0.0, 0.1, 0.995])
q_targ = q_targ / np.linalg.norm(q_targ)

print("Target:", q_targ)
for i in range(10):
    q_curr = compute_orientation(q_curr, q_targ)
    print(f"Step {i}: {q_curr}")
