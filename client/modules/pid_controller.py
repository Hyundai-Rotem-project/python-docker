class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # 비례 항
        self.ki = ki  # 적분 항
        self.kd = kd  # 미분 항
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt=1.0):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output