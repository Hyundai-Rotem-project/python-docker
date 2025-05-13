from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import cv2, os
from StereoImageFilter import StereoImageFilter
from sklearn.linear_model import LinearRegression

"""
    
"""
class DistAngle:
    def __init__(self, left_dir, right_dir, log_path, camera_intrinsic=None):
        """
        log_data : pandas DataFrame
        log_data columns :
            'Time', 'Distance', 'Player_Pos_X', 'Player_Pos_Y', 'Player_Pos_Z',
            'Player_Speed', 'Player_Health', 'Player_Turret_X', 'Player_Turret_Y',
            'Player_Body_X', 'Player_Body_Y', 'Player_Body_Z', 'TurretCam_X',
            'TurretCam_Y', 'TurretCam_Z', 'StereoL_X', 'StereoL_Y', 'StereoL_Z',
            'StereoL_Roll', 'StereoL_Pitch', 'StereoL_Yaw', 'StereoR_X',
            'StereoR_Y', 'StereoR_Z', 'StereoR_Roll', 'StereoR_Pitch',
            'StereoR_Yaw', 'Enemy_Pos_X', 'Enemy_Pos_Y', 'Enemy_Pos_Z',
            'Enemy_Speed', 'Enemy_Health', 'Enemy_Turret_X', 'Enemy_Turret_Y',
            'Enemy_Body_X', 'Enemy_Body_Y', 'Enemy_Body_Z'],
        """
        self.left_dir = left_dir
        self.right_dir = right_dir

        #필터링된 log_data, detection_dict
        filter = StereoImageFilter(left_dir, right_dir, log_path)
        log_data, detection_left, detection_right = filter.get_result()
        self.log_data = log_data
        
        self.detection_left = detection_left
        self.detection_right = detection_right
        """
        self.time = log_data[['Time']]
        self.distance = log_data[['Distance']]
        # X:좌 Z:전 Y:상
        self.player_pos = log_data[['Player_Pos_X','Player_Pos_Y','Player_Pos_Z']]
        # X:Yaw Y:Pitch Z:Roll, 각각 Y,X,Z축 기준으로 회전축의 +방향에서 바라봤을 때 반시계방향이 양의 회전방향향
        self.player_rot = log_data[['Player_Body_X','Player_Body_Y','Player_Body_Z']]
        #Stereo Position&Rotation
        self.stereoL_pos = log_data[['StereoL_X','StereoL_Y','StereoL_Z']]
        self.stereoL_rot = log_data[['StereoL_Yaw','StereoL_Pitch','StereoL_Roll']]
        self.stereoR_pos = log_data[['StereoR_X','StereoR_Y','StereoR_Z']]
        self.stereoR_rot = log_data[['StereoR_Yaw','StereoR_Pitch','StereoR_Roll']]

        self.enemy_pos = log_data[['Enemy_Body_X', 'Enemy_Body_Y', 'Enemy_Body_Z']]
        """
        # 카메라 내부 파라미터가 없는 경우 기본값 사용
        if camera_intrinsic is None:
            # 이미지 크기 512x512 기준으로 설정
            fx = 1200  # 초첨 거리 (x축)
            fy = 1200  # 초첨 거리 (y축)
            cx = 256   # 주점 x 좌표
            cy = 256   # 주점 y 좌표
            self.camera_intrinsic = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        
    def pixel_to_ray(bx, by, camera_pos, camera_rot, camera_intrinsic=None):
        """
        이미지 상의 픽셀 좌표를 3D 광선으로 변환합니다.

        Args:
            bx, by: 이미지 상의 객체 중심 좌표 (픽셀)
            camera_pos: 카메라 위치 (x, y, z)
            camera_rotation: 카메라 방향 (roll, pitch, yaw) (라디안)
            camera_intrinsic: 카메라 내부 파라미터 매트릭스 (없으면 기본값 사용)

        Returns:
            origin: 광선의 시작점 (카메라 위치)
            direction: 광선의 방향 벡터 (정규화됨)
        """
        # 카메라 내부 파라미터가 없는 경우 기본값 사용
        if camera_intrinsic is None:
            # 이미지 크기 1920x1080 기준으로 임시 설정
            fx = 1200  # 초첨 거리 (x축)
            fy = 1200  # 초첨 거리 (y축)
            cx = 256   # 주점 x 좌표
            cy = 256   # 주점 y 좌표
            camera_intrinsic = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])

        # 픽셀 좌표를 정규화된 카메라 좌표로 변환
        normalized_coords = np.linalg.inv(camera_intrinsic) @ np.array([bx, by, 1])

        # 카메라 좌표계에서의 광선 방향
        ray_camera = normalized_coords / np.linalg.norm(normalized_coords)

        # 카메라 회전 행렬 계산 (roll, pitch, yaw를 회전 행렬로 변환)
        rotation = R.from_euler('xyz', camera_rot)
        rotation_matrix = rotation.as_matrix()

        # 카메라 좌표계의 광선을 월드 좌표계로 변환
        ray_direction = rotation_matrix @ ray_camera

        # 정규화
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        return camera_pos, ray_direction

    def find_closest_point_between_rays(origin1, direction1, origin2, direction2):
        """
        두 광선 사이의 최근접점을 찾습니다.

        Args:
            origin1, direction1: 첫 번째 광선의 시작점과 방향
            origin2, direction2: 두 번째 광선의 시작점과 방향

        Returns:
            midpoint: 두 광선 사이의 최근접점 (3D 공간 좌표)
            distance: 두 광선 사이의 최소 거리
        """
            # 광선 방향 벡터가 단위 벡터인지 확인
        direction1 = direction1 / np.linalg.norm(direction1)
        direction2 = direction2 / np.linalg.norm(direction2)

        # 연립방정식을 풀기 위한 행렬 설정
        A = np.array([
            [np.dot(direction1, direction1), -np.dot(direction1, direction2)],
            [np.dot(direction1, direction2), -np.dot(direction2, direction2)]
        ])  

        # 원점 차이 계산
        delta = origin1 - origin2

        # 연립방정식의 우변 설정
        b = np.array([
            np.dot(direction1, delta),
            np.dot(direction2, delta)
        ])

        # 연립방정식 풀이
        try:
            t1, t2 = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 행렬이 특이해서 해가 없는 경우 (광선이 평행한 경우)
            return None, float('inf')

        # 각 광선 위의 점 계산
        point1 = origin1 + t1 * direction1
        point2 = origin2 + t2 * direction2

        # 두 점 사이의 중간점을 최근접점으로 설정
        midpoint = (point1 + point2) / 2

            # 두 광선 사이의 최소 거리 계산
        distance = np.linalg.norm(point1 - point2)
    
        return midpoint, distance

    def compute_3d_position(self, stereoL_pos, stereoL_rot, stereoR_pos, stereoR_rot,
                           bx_left, by_left, bx_right, by_right, camera_intrinsic=None):
        """
        스테레오 이미지에서 객체의 3D 좌표를 계산합니다.

        Args:
            player_pos: 플레이어 위치 (x, y, z)
            stereoL_pos: 왼쪽 카메라 위치 (x, y, z)
            stereoL_rot: 왼쪽 카메라 회전 (roll, pitch, yaw)
            stereoR_pos: 오른쪽 카메라 위치 (x, y, z)
            stereoR_rot: 오른쪽 카메라 회전 (roll, pitch, yaw)
            bx_left, by_left: 왼쪽 이미지에서의 객체 중심 좌표
            bx_right, by_right: 오른쪽 이미지에서의 객체 중심 좌표
            camera_intrinsic: 카메라 내부 파라미터 (선택사항)

        Returns:
            3D 좌표 (x, y, z)
        """
        # 각 카메라에서 광선 계산
        origin_left, direction_left = self.pixel_to_ray(bx_left, by_left, stereoL_pos, stereoL_rot)
        origin_right, direction_right = self.pixel_to_ray(bx_right, by_right, stereoR_pos, stereoR_rot)

        # 두 광선의 최근접점 계산
        midpoint, distance = self.find_closest_point_between_rays(origin_left, direction_left, origin_right, direction_right)

        # 거리가 너무 크면 삼각측량 실패로 간주
        if distance > 1.0:  # 임계값은 상황에 맞게 조정
            print(f"Warning: 두 광선 사이의 거리가 너무 큽니다: {distance}")

        return midpoint
    
    def image_center_dict(self, detect_dict):
        image_center_dict = {}
        for time, value in detect_dict.items():
            x1,y1,x2,y2 = value[0]['bbox']
            bx = (x1+x2)/2
            by = (y1+y2)/2
            image_center_dict.update({time:[bx,by]})
        return image_center_dict
    
    def estimated_dist_angle(self,player_pos, stereoL_pos, stereoL_rot, stereoR_pos, stereoR_rot,
                           bx_left, by_left, bx_right, by_right, camera_intrinsic=None):
        """
        return:
            estimated_dist : 객체까지 계산된 거리 float
            estimated_dir : 객체 방향 벡터 [x,y,z]
        """
        estimated_objetect_position = self.compute_3d_position(stereoL_pos, stereoL_rot, stereoR_pos, stereoR_rot,
                           bx_left, by_left, bx_right, by_right, camera_intrinsic=None)
        estimated_dist = np.linalg.norm(estimated_objetect_position-player_pos)
        estimated_dir = (estimated_objetect_position-player_pos)/estimated_dist
        return estimated_dist, estimated_dir
    
    def get_disparity(self):
        """

        """
        time=self.log_data['Time']
        left_dir = self.left_dir
        right_dir = self.right_dir
        disparity_dict = {}
        for time in time:
            img_L = cv2.imread(os.path.join(left_dir, f'{time}.png'),cv2.IMREAD_GRAYSCALE)
            img_R = cv2.imread(os.path.join(right_dir, f'{time}.png'),cv2.IMREAD_GRAYSCALE)

            stereo = cv2.StereoSGBM_create(minDisparity=0,
                               numDisparities=64,
                               blockSize=9,
                               P1=8 * 3 * 9 ** 2,
                               P2=32 * 3 * 9 ** 2,
                               disp12MaxDiff=1,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32)
            disparity = stereo.compute(img_L, img_R).astype(np.float32) / 16.0
            disparity_dict.update({time:disparity})
        return disparity_dict
    
    def get_pixel_values(self, image_list, coordinates_list):
        """
        512x512 이미지 리스트와 좌표 리스트를 입력받아
        각 좌표에 해당하는 이미지 값을 추출하여 새로운 리스트로 반환합니다.

        Args:
            image_list (list of list): 512x512 크기의 이미지 데이터를 담고 있는 2차원 리스트.
            coordinates_list (list of list): 이미지 내 좌표 [[x1, y1], [x2, y2], ...] 형태의 리스트.

        Returns:
            list: 좌표에 대응하는 이미지 값들을 담은 새로운 리스트.
        """
        pixel_values = []
        for image in image_list:
            for x, y in coordinates_list:
                x=int(round(x,0))
                y=int(round(y,0))
                pixel_values.append(image[y][x])  # 이미지 리스트는 image[y][x]로 접근
        return pixel_values
    
    def get_table_for_regression(self):
        """
        기준:"Time"
        X={
            Player_Pos
            StereoL_Pos
            StereoR_Pos
            StereoL_Rot
            box_size
            bx_left, by_left
            bx_right, by_right
            disparity_at_the_point
            estimated_dist, estimated_dir
            speed
        }
        y = real_dist, estimated_dir
        """
        log_data = self.log_data

        time = self.time

        #[x,y,z]
        player_pos = self.player_pos
        enemy_pos = log_data[['Enemy_Pos_X','Enemy_Pos_Y','Enemy_Pos_Z']]
        #[x,y,z]
        stereoL_pos = self.stereoL_pos
        stereoR_pos = self.stereoR_pos
        #[yaw,pitch,roll]
        stereoL_rot = self.stereoL_rot
        stereoR_rot = self.stereoR_rot
        
        detection_left = self.detection_left
        detection_right = self.detection_right


        box_size_left = []
        for values in detection_left.values():
            x1, y1, x2, y2 = values[0]['bbox']
            box_size_left.append((x2 - x1) * (y2 - y1))

        box_size_right = []
        for values in detection_right.values():
            x1, y1, x2, y2 = values[0]['bbox']
            box_size_right.append((x2 - x1) * (y2 - y1))
            
        

        #{time:[bx,by]}
        image_center_dict_left = list(self.image_center_dict(detection_left).values())
        bx_left=[] 
        by_left=[]
        for coord in image_center_dict_left:
            bx_left.append(coord[0])
            by_left.append(coord[1]) 
        image_center_dict_right = list(self.image_center_dict(detection_left).values())
        bx_right=[] 
        by_right=[]
        for coord in image_center_dict_right:
            bx_right.append(coord[0])
            by_right.append(coord[1]) 

        #{time:disparity} -> 박스안의 disparity 값
        disparity_dict = self.get_disparity()
        disparity_image=list(disparity_dict.values())
        disparity_pixel_value = self.get_pixel_values(disparity_image, image_center_dict_left)
       

        #float, [x,y,z] 단위 벡터
        estimated_dist = []
        estimated_dir = []
        for p_pos, l_pos, l_rot, r_pos, r_rot, bx_l, by_l, bx_r, by_r in zip(
                player_pos, stereoL_pos, stereoL_rot, stereoR_pos, stereoR_rot,
                bx_left, by_left, bx_right, by_right):
            
            dist, dir = self.estimated_dist_angle(
                p_pos, l_pos, l_rot, r_pos, r_rot,
                bx_l, by_l, bx_r, by_r
            )

            estimated_dist.append(dist)
            estimated_dir.append(dir)

        real_dist = log_data['Distance']
        real_dir = (player_pos-enemy_pos)/real_dist
        real_dir = real_dir.to_list()

        player_velocity = []

        for i in range(1, len(player_pos)):
            delta_pos = player_pos[i] - player_pos[i-1]
            delta_time = time[i] - time[i-1]

        # 시간 간격이 0이 아닌 경우에만 속도 계산
            if delta_time != 0:
                velocity = delta_pos / delta_time
                player_velocity.append(velocity)
            else:
        # 시간 간격이 0인 경우, 속도를 0 또는 다른 적절한 값으로 처리
                player_velocity.append(0) # 예시: 속도를 0으로 처리

        # 첫 번째 속도 값 처리 (이전 값이 없으므로 0으로 초기화하거나 다른 방식으로 처리)
        if player_velocity:
            player_velocity.insert(0, 0) # 예시: 첫 번째 속도를 0으로 초기화
        elif time:
            player_velocity.append(0) # 데이터가 하나인 경우 속도는 0

        log_data_X =pd.DataFrame()
        log_data_X = log_data[['Player_Pos_X', 'Player_Pos_Y', 'Player_Pos_Z', 'StereoL_X', 'StereoL_Y', 'StereoL_Z','StereoL_Roll', 'StereoL_Pitch', 'StereoL_Yaw', 'StereoR_X','StereoR_Y', 'StereoR_Z']]
        log_data_X['bx_left'] = bx_left
        log_data_X['by_left'] = by_left
        log_data_X['bx_right'] = bx_right
        log_data_X['by_right'] = by_right
        log_data_X['box_size_left'] = box_size_left
        log_data_X['box_size_right'] = box_size_right
        log_data_X['disparity'] = disparity_pixel_value
        log_data_X['estimated_distance'] = estimated_dist
        log_data_X['est_dir_x'] = [dir[0] for dir in estimated_dir]
        log_data_X['est_dir_y'] = [dir[1] for dir in estimated_dir]
        log_data_X['est_dir_z'] = [dir[2] for dir in estimated_dir]
        log_data_X['Player_Speed'] = player_velocity

        log_data_y = pd.DataFrame()
        log_data_y['distance'] = log_data['Distance']
        log_data_y['dir_x'] = real_dir[0]
        log_data_y['dir_y'] = real_dir[1]
        log_data_y['dir_z'] = real_dir[2]

        return log_data_X, log_data_y
    
def main():
    left_folder = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\capture_images\\L"
    right_folder = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\capture_images\\R"
    log_path = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\log_data\\tank_info_log.txt"
    
    DistAngle1 = DistAngle(left_folder,right_folder,log_path)

    log_X, log_y = DistAngle1.get_table_for_regression()
    print(log_X,log_y)

if __name__=='__main__':
    main()

        






