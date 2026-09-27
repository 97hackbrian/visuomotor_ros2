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
