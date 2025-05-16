import math

class Ballistics:
    def __init__(self, context):
        self.context = context

    def _calculation_of_barrel_angle_by_distance(self):
        # 원 회귀식; y=0.373x2+5.914x+41.24; y: distance, x: barrel_degree
        # 적과의 거리가 사정거리 내인지 확인할 것
        distance = self.context.shared_data["distance"]
        if self.context.EFFECTIVE_MIN_RANGE <= distance <= self.context.EFFECTIVE_MAX_RANGE:
            # 포신 각도를 회귀식을 통해 구하기기
            if not (20.995 <= distance <= 137.68):
                raise ValueError("Distance is outside the inverse function's domain [20.995, 137.68].")

            # 원 회귀식의 역함수
            discriminant = 1.492 * distance - 24.564784 # 기존 회기식에 대한 역함수의 상수를를 -26.564784에서 -24.564784로 변경(더 높은 사거리 선정을 위해해)
            if discriminant < 0:
                raise ValueError("Discriminant is negative. No real solutions exist.")

            barrel_angle_deg = (-5.914 + math.sqrt(discriminant)) / 0.746  # In degrees
            if not (-5.0 + 1e-6 <= barrel_angle_deg <= 10.0 + 1e-6):
                raise ValueError("Calculated barrel angle is outside the range [-5, 10].")

            # Convert barrel angle to radians (for error calculation)
            barrel_angle = barrel_angle_deg * math.pi / 180

            # Calculate barrel angle error
            current_turret_angle_rad = self.context.shared_data["playerTurretY"] * math.pi / 180
            barrel_angle_error = current_turret_angle_rad - barrel_angle
            barrel_angle_error = math.atan2(math.sin(barrel_angle_error), math.cos(barrel_angle_error))
            
            return barrel_angle, barrel_angle_error
        else:
            raise ValueError("Distance exceeds effective range")

    def _calculation_of_barrel_angle_by_distance_with_delta_h(self):
        # 원 회귀식: theta = 0.373x^2 + 5.914x + 41.24 (theta: barrel angle in degrees, x: distance)
        # 높이 차이 delta_h를 고려한 새로운 포신 각도 계산
        # 적과의 거리가 사정거리 내인지 확인
        distance = self.context.shared_data["distance"]
        delta_h = self.context.shared_data["enemyPos"]["y"] - self.context.shared_data["playerPos"]["y"]  # delta_h가 없으면 0으로 설정
        self.barrel_angle, self.barrel_angle_error = self._calculation_of_barrel_angle_by_distance()
        
        if self.context.EFFECTIVE_MIN_RANGE <= distance <= self.context.EFFECTIVE_MAX_RANGE:
            # 포신 각도를 회귀식과 delta_h를 통해 구하기
            theta_old_rad = self.barrel_angle
            
            # theta_new = arctan(tan(theta_old) + delta_h / distance)
            tan_theta_new = math.tan(theta_old_rad) + delta_h / distance
            barrel_angle_deg = math.atan(tan_theta_new) * 180 / math.pi  # 도 단위로 변환
            
            # 포신 각도 범위 확인
            if not (-5.0 + 1e-6 <= barrel_angle_deg <= 10.0 + 1e-6):
                print("barrel_angle: ", barrel_angle_deg)
                raise ValueError("Calculated barrel angle is outside the range [-5, 10].")

            # Convert barrel angle to radians (for output)
            barrel_angle = barrel_angle_deg * math.pi / 180

            # Calculate barrel angle error
            current_turret_angle_rad = self.context.shared_data["playerTurretY"] * math.pi / 180
            barrel_angle_error = current_turret_angle_rad - barrel_angle
            barrel_angle_error = math.atan2(math.sin(barrel_angle_error), math.cos(barrel_angle_error))
            print("barrel_angle: ",barrel_angle, "barrel_angle_error: ", barrel_angle_error)
            
            return barrel_angle, barrel_angle_error
        else:
            raise ValueError("Distance exceeds effective range")