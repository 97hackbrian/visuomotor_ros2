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
        
        self.virtual_euler = None
        self.integral_euler = np.zeros(3)
        self.prev_error_euler = np.zeros(3)
        
    def reset(self):
        self.virtual_setpoint = None
        self.virtual_euler = None
        self.integral_euler = np.zeros(3)
        self.prev_error_euler = np.zeros(3)
        
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
            
        error = target_pos - self.virtual_setpoint
        
        p_term = self.kp * error
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
        if self.virtual_setpoint is None or target_quat is None:
            # Si el PID no ha iniciado o no hay target explícito, no rotamos
            return current_quat
            
        # Determinar velocidad angular máxima
        max_rot_step = self.max_speed_rad_s / self.sampling_rate_hz
        
        # Calcular SLERP
        q1 = np.array(current_quat)
        q2 = np.array(target_quat)
        
        dot = np.sum(q1 * q2)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
            
        if dot > 0.9995:
            # Linear interpolation para ángulos muy pequeños
            res = q1 + self.kp_rot * (q2 - q1)
            res = res / np.linalg.norm(res)
            return res
            
        theta_0 = np.arccos(dot)
        
        # El paso angular deseado es theta_0 * kp_rot, pero lo limitamos a max_rot_step
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
        return res / np.linalg.norm(res)
