import numpy as np

class CartesianPID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.05, max_speed_m_s=0.15, sampling_rate_hz=20.0,
                 kp_rot=1.5, ki_rot=0.0, kd_rot=0.05, max_speed_rad_s=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_speed_m_s = max_speed_m_s
        self.sampling_rate_hz = sampling_rate_hz
        
        self.kp_rot = kp_rot
        self.ki_rot = ki_rot
        self.kd_rot = kd_rot
        self.max_speed_rad_s = max_speed_rad_s
        
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.virtual_setpoint = None
        self.virtual_quat = None
        
        self.virtual_euler = None
        self.integral_euler = np.zeros(3)
        self.prev_error_euler = np.zeros(3)
        
    def reset(self):
        self.virtual_setpoint = None
        self.virtual_quat = None
        self.virtual_euler = None
        self.integral_euler = np.zeros(3)
        self.prev_error_euler = np.zeros(3)
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        
    def update_params(self, kp, ki, kd, max_speed_m_s, kp_rot=None, ki_rot=None, kd_rot=None, max_speed_rad_s=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_speed_m_s = max_speed_m_s
        
        if kp_rot is not None: self.kp_rot = kp_rot
        if ki_rot is not None: self.ki_rot = ki_rot
        if kd_rot is not None: self.kd_rot = kd_rot
        if max_speed_rad_s is not None: self.max_speed_rad_s = max_speed_rad_s
        
    def compute(self, current_pos, target_pos):
        """
        Genera un setpoint virtual que se mueve hacia el target a velocidad constante (a 100Hz),
        y luego decelera suavemente usando control proporcional, integral y derivativo.
        Garantiza que la velocidad física sea EXACTAMENTE max_speed_m_s sin tartamudeos.
        """
        if self.virtual_setpoint is None:
            self.virtual_setpoint = current_pos.copy()
            
        # [CRITICAL] Prevent the virtual setpoint from running away if the physical robot is stuck
        dist_to_robot = np.linalg.norm(self.virtual_setpoint - current_pos)
        if dist_to_robot > 0.05: # Max 5cm lead
            self.virtual_setpoint = current_pos + ((self.virtual_setpoint - current_pos) / dist_to_robot) * 0.05
            
        error = target_pos - self.virtual_setpoint
        
        effective_kp = min(self.kp, 1.0)
        p_term = effective_kp * error
        self.integral += error * (1.0 / self.sampling_rate_hz)
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.prev_error) * self.sampling_rate_hz
        self.prev_error = error
        
        step = p_term + i_term + d_term
        
        max_step = self.max_speed_m_s / self.sampling_rate_hz
        norm = np.linalg.norm(step)
        if norm > max_step:
            step = (step / norm) * max_step
            
        self.virtual_setpoint += step
        
        control_output = self.virtual_setpoint - current_pos
        real_error = target_pos - current_pos
            
        return control_output, real_error

    def compute_euler(self, current_euler, target_euler):
        if self.virtual_euler is None:
            self.virtual_euler = np.array(current_euler, dtype=np.float64)
            
        current_euler = np.array(current_euler, dtype=np.float64)
        target_euler = np.array(target_euler, dtype=np.float64)
            
        # Error angular (distancia más corta)
        error = np.arctan2(np.sin(target_euler - self.virtual_euler), np.cos(target_euler - self.virtual_euler))
        
        p_term = self.kp_rot * error
        self.integral_euler += error * (1.0 / self.sampling_rate_hz)
        i_term = self.ki_rot * self.integral_euler
        d_term = self.kd_rot * (error - self.prev_error_euler) * self.sampling_rate_hz
        self.prev_error_euler = error
        
        step = p_term + i_term + d_term
        
        # Limitar velocidad angular
        max_rot_step = self.max_speed_rad_s / self.sampling_rate_hz
        norm_step = np.linalg.norm(step)
        if norm_step > max_rot_step:
            step = (step / norm_step) * max_rot_step
            
        self.virtual_euler += step
        # Normalizar a -pi, pi
        self.virtual_euler = np.arctan2(np.sin(self.virtual_euler), np.cos(self.virtual_euler))
        
        control_output = np.arctan2(np.sin(self.virtual_euler - current_euler), np.cos(self.virtual_euler - current_euler))
        real_error = np.arctan2(np.sin(target_euler - current_euler), np.cos(target_euler - current_euler))
        
        return control_output, real_error

    def compute_orientation(self, current_quat, target_quat):
        """
        Interpola suavemente la orientación actual hacia la deseada.
        current_quat, target_quat: [x, y, z, w]
        """
        if not hasattr(self, 'virtual_quat') or self.virtual_quat is None:
            self.virtual_quat = np.array(current_quat)
            
        # [CRITICAL] Prevent virtual_quat from running away from the physical robot
        dot_curr = np.clip(np.abs(np.sum(self.virtual_quat * np.array(current_quat))), -1.0, 1.0)
        ang_dist_to_robot = 2.0 * np.arccos(dot_curr)
        if ang_dist_to_robot > 0.15: # Max ~8.5 degrees lead
            # Pull virtual_quat back closer to current_quat
            t_pull = 0.15 / ang_dist_to_robot
            
            cq = np.array(current_quat)
            vq = self.virtual_quat
            if np.sum(cq * vq) < 0.0:
                cq = -cq
                
            sin_theta_0 = np.sin(ang_dist_to_robot / 2.0)
            theta = (ang_dist_to_robot / 2.0) * t_pull
            sin_theta = np.sin(theta)
            dot_v = np.sum(cq * vq)
            
            s0 = np.cos(theta) - dot_v * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0
            self.virtual_quat = (s0 * cq) + (s1 * vq)
            self.virtual_quat /= np.linalg.norm(self.virtual_quat)
            
        if target_quat is None:
            return current_quat
            
        # Determinar velocidad angular máxima
        max_rot_step = self.max_speed_rad_s / self.sampling_rate_hz
        
        # Calcular SLERP desde el virtual_quat (no desde current_quat)
        q1 = self.virtual_quat
        q2 = np.array(target_quat)
        
        dot = np.sum(q1 * q2)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
            
        if dot > 0.9995:
            # Linear interpolation para ángulos muy pequeños (sin overshoot)
            effective_kp = min(self.kp_rot, 1.0)
            res = q1 + effective_kp * (q2 - q1)
            res = res / np.linalg.norm(res)
            self.virtual_quat = res
            return res
            
        theta_0 = np.arccos(dot)
        
        # El paso angular deseado es theta_0 * kp_rot, pero lo limitamos a max_rot_step y evitamos overshoot
        effective_kp = min(self.kp_rot, 1.0)
        step_theta = theta_0 * effective_kp
        if step_theta > max_rot_step:
            t = max_rot_step / theta_0
        else:
            t = effective_kp
            
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        res = (s0 * q1) + (s1 * q2)
        res = res / np.linalg.norm(res)
        self.virtual_quat = res
        return res
