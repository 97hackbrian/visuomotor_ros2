import numpy as np

class CartesianPID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.05, max_speed_m_s=0.15, sampling_rate_hz=20.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_speed_m_s = max_speed_m_s
        self.sampling_rate_hz = sampling_rate_hz
        
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.virtual_setpoint = None
        
    def reset(self):
        self.virtual_setpoint = None
        
    def update_params(self, kp, ki, kd, max_speed_m_s):
        self.kp = kp
        self.max_speed_m_s = max_speed_m_s
        
    def compute(self, current_pos, target_pos):
        """
        Genera un setpoint virtual que se mueve hacia el target a velocidad constante (a 100Hz),
        y luego decelera suavemente usando control proporcional.
        Garantiza que la velocidad física sea EXACTAMENTE max_speed_m_s sin tartamudeos.
        """
        if self.virtual_setpoint is None:
            self.virtual_setpoint = current_pos.copy()
            
        error = target_pos - self.virtual_setpoint
        
        step = self.kp * error
        
        max_step = self.max_speed_m_s / self.sampling_rate_hz
        norm = np.linalg.norm(step)
        if norm > max_step:
            step = (step / norm) * max_step
            
        self.virtual_setpoint += step
        
        control_output = self.virtual_setpoint - current_pos
        real_error = target_pos - current_pos
            
        return control_output, real_error

    def compute_orientation(self, current_quat, target_quat):
        """
        Interpola suavemente la orientación actual hacia la deseada.
        current_quat, target_quat: [x, y, z, w]
        """
        if self.virtual_setpoint is None:
            # Si el PID de posición no ha iniciado, no hacemos rotación
            return current_quat
            
        # Determinar velocidad angular máxima basada en max_speed
        # Usamos una relación empírica (ej. 1 m/s = 2 rad/s)
        max_rot_step = (self.max_speed_m_s * 2.0) / self.sampling_rate_hz
        
        # Calcular SLERP
        q1 = np.array(current_quat)
        q2 = np.array(target_quat)
        
        dot = np.sum(q1 * q2)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
            
        if dot > 0.9995:
            # Linear interpolation para ángulos muy pequeños
            res = q1 + self.kp * (q2 - q1)
            res = res / np.linalg.norm(res)
            return res
            
        theta_0 = np.arccos(dot)
        
        # El paso angular deseado es theta_0 * kp, pero lo limitamos a max_rot_step
        step_theta = theta_0 * self.kp
        if step_theta > max_rot_step:
            t = max_rot_step / theta_0
        else:
            t = self.kp
            
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        res = (s0 * q1) + (s1 * q2)
        return res / np.linalg.norm(res)
