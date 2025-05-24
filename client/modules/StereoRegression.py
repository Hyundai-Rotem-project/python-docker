from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import cv2, os
from modules.StereoImageFilter import StereoImageFilter
import joblib

class DistCalculator:
    def __init__(self, camera_intrinsic=None):        
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
    def get_obj_dist_pitch_yaw(self, bx, by, camera_pos, camera_rot):
        u = (bx-256)/256
        v = -(by-256)/256
        roll, pitch, yaw = camera_rot
        return

    def pixel_to_ray(self, bx, by, camera_pos, camera_rot, camera_intrinsic=None):
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

        #if len(list(camera_rot)) != 3:
        #    raise ValueError("입력 회전 벡터는 반드시 3개의 값 (roll, pitch, yaw)이 필요합니다.")

        # 카메라 내부 파라미터가 없는 경우 기본값 사용
        if camera_intrinsic is None:
            # 이미지 크기 512x512 기준으로 설정
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

        # 카메라 회전 행렬 계산 (yaw, pitch, roll를 회전 행렬로 변환)
        camera_rot_rad = np.deg2rad(np.array(camera_rot) * -1) 
        rotation = R.from_euler('xyz', camera_rot)
        rotation_matrix = rotation.as_matrix()

        # 카메라 좌표계의 광선을 월드 좌표계로 변환
        ray_direction = rotation_matrix @ ray_camera

        # 정규화
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        #axis_yaw = camera_pos[2]
        #axis_roll = camera_pos[1]
        #camera_pos[1] = axis_roll
        #camera_pos[2] = axis_yaw
        return np.array(camera_pos), ray_direction
    
    def find_closest_point_between_rays(self, origin1, direction1, origin2, direction2):
        """
        두 광선 사이의 최근접점을 찾습니다.
        input = single value

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

        # 두 점 사이의 중간점을 최근접점으로 설정 다시 unity좌표계로로
        midpoint = (point1 + point2) / 2
        #axis_yaw = midpoint[2]
        #axis_roll = midpoint[1]
        #midpoint[1] = axis_yaw
        #midpoint[2] = axis_roll
        # 두 광선 사이의 최소 거리 계산
        distance = np.linalg.norm(point1 - point2)


        return midpoint, distance
    

    def compute_3d_position(self,bx_left, by_left, bx_right, by_right, stereoL_pos, stereoL_rot, stereoR_pos, stereoR_rot, camera_intrinsic=None):
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

        # 각 카메라에서 광선 계산 / 일반적인 좌표계로 변환된 좌표값
        origin_left, direction_left = self.pixel_to_ray(bx_left, by_left, stereoL_pos, stereoL_rot)
        origin_right, direction_right = self.pixel_to_ray(bx_right, by_right, stereoR_pos, stereoR_rot)
        
        # 두 광선의 최근접점 계산 시뮬레이터상 좌표로 출력
        midpoint, distance = self.find_closest_point_between_rays(origin_left, direction_left, origin_right, direction_right)

        # 거리가 너무 크면 삼각측량 실패로 간주
        #if distance > 1.0:  # 임계값은 상황에 맞게 조정
        #    print(f"Warning: 두 광선 사이의 거리가 너무 큽니다: {distance}")

        return midpoint
    

class StereoPreprocess:
    def __init__(self, left_dir, right_dir, log_path, model_path, camera_intrinsic=None):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.log_path = log_path
        self.model_path = model_path
        #필터링된 log_data, detection_dict -> Time 기준 정렬렬
        filter = StereoImageFilter(left_dir, right_dir, log_path, self.model_path)
        log_data, detection_left, detection_right = filter.get_result()
        log_data = log_data.sort_values(by='Time')
        self.log_data = log_data
        self.detection_left = detection_left
        self.detection_right = detection_right
        print('detection left length:',len(detection_left))

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
        
    def detect_dict_to_center_list(self, detect_dict):
        def box_center_dict(detect_dict):
            box_center_dict = {}
            for time, value in detect_dict.items():
                x1,y1,x2,y2 = value[0]['bbox']
                bx = (x1+x2)/2
                by = (y1+y2)/2
                box_size = (x2-x1)*(y2-y1)
                box_center_dict[time]={'box_center':[bx,by,box_size]}
            return box_center_dict
    
        def get_bxby(box_center_dict):
            box_center_list = sorted(box_center_dict.items())
            bx = []
            by = []
            boxes_size = []
            center_list=[]
            idx = len(box_center_list)
            for i in range(0,idx): 
                [[box_center_x, box_center_y, box_size]] = box_center_list[i][1].values()
                bx.append(box_center_x)
                by.append(box_center_y)
                boxes_size.append(box_size)
                center_list.append([box_center_x,box_center_y]) 
            return bx, by, boxes_size, center_list
        
        center_dict = box_center_dict(detect_dict)
        bx, by, boxes_size, center_list = get_bxby(center_dict)
        return bx, by, boxes_size, center_list


    def estimated_dist_angles(self, player_pos, estimated_object_pos, camera_intrinsic=None):
        """
        return:
            estimated_dist : 객체까지 계산된 거리 float
            estimated_dir : 객체 방향 벡터 [x,y,z]
        """
        positions = pd.concat([player_pos, estimated_object_pos], axis=1)
        estimated_dist=[]
        estimated_dir=[]
        for idx, row in positions.iterrows():
            pl_pos = np.array([row['Player_Pos_X'], row['Player_Pos_Y'], row['Player_Pos_Z']])
            obj_pos = np.array([row['X'],row['Y'],row['Z']])
            dist = np.linalg.norm(obj_pos-pl_pos)
            dir = (obj_pos-pl_pos)/dist
            estimated_dist.append(dist)
            estimated_dir.append(dir)
        estimated_dist = pd.DataFrame(estimated_dist,columns=['estimated_distance'])
        estimated_dir = pd.DataFrame(estimated_dir,columns=['X','Y','Z'])
        return estimated_dist, estimated_dir
    
    def get_disparity(self, log_data, left_dir, right_dir):
        time = log_data['Time']
        left_dir = left_dir
        right_dir = right_dir
        disparity_dict = {}
        for time in time:
            img_L = cv2.imread(os.path.join(left_dir, f'{time:.2f}.png'),cv2.IMREAD_GRAYSCALE)
            img_R = cv2.imread(os.path.join(right_dir, f'{time:.2f}.png'),cv2.IMREAD_GRAYSCALE)

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
        disparity_list = sorted(disparity_dict.items())
        return disparity_list
    
    def get_pixel_values(self, disparity_list, coordinates_list):
        """
        512x512 이미지 리스트와 좌표 리스트를 입력받아
        각 좌표에 해당하는 이미지 값을 추출하여 새로운 리스트로 반환합니다.

        Args:
            image_list (list of ndarray): 512x512 크기의 이미지 데이터를 담고 있는 2차원 리스트.
            coordinates_list (list of list): 이미지 내 좌표 [[x1, y1], [x2, y2], ...] 형태의 리스트.

        Returns:
            list: 좌표에 대응하는 이미지 값들을 담은 새로운 리스트.
        """
        pixel_values = []
        list_len = len(coordinates_list)


        for i in range(0,list_len):
            x,y = coordinates_list[i]
            image=disparity_list[i][1]
            x=int(round(x,0))
            y=int(round(y,0))
            pixel_value = image[y][x]
            if pixel_value >= 0:
                pixel_values.append(pixel_value)  # 이미지 리스트는 image[y][x]로 접근
            else:
                pixel_value=0
                pixel_values.append(pixel_value)
        return pixel_values
    
    def get_velocity(self, log_data):
        player_pos = log_data[['Player_Pos_X', 'Player_Pos_Y', 'Player_Pos_Z']]
        time_diff = log_data['Time'].diff()
        pos_diff = player_pos.diff()

        # 시간 변화량이 0인 경우 NaN으로 처리하여 나눗셈 오류 방지
        velocity = pos_diff.div(time_diff, axis=0).fillna(0)
    
        # 첫 번째 행의 속도는 계산할 수 없으므로 0으로 채움
        velocity.iloc[0] = 0
        return velocity

    def get_log_data(self):
        '''
        return log_data (pd.DataFrame) :
            ['Time', 'Distance', 'Player_Pos_X', 'Player_Pos_Y', 'Player_Pos_Z',
            'Player_Speed', 'Player_Health', 'Player_Turret_X', 'Player_Turret_Y',
            'Player_Body_X', 'Player_Body_Y', 'Player_Body_Z', 'TurretCam_X',
            'TurretCam_Y', 'TurretCam_Z', 'StereoL_X', 'StereoL_Y', 'StereoL_Z',
            'StereoL_Roll', 'StereoL_Pitch', 'StereoL_Yaw', 'StereoR_X',
            'StereoR_Y', 'StereoR_Z', 'StereoR_Roll', 'StereoR_Pitch',
            'StereoR_Yaw', 'Enemy_Pos_X', 'Enemy_Pos_Y', 'Enemy_Pos_Z',
            'Enemy_Speed', 'Enemy_Health', 'Enemy_Turret_X', 'Enemy_Turret_Y',
            'Enemy_Body_X', 'Enemy_Body_Y', 'Enemy_Body_Z', 'bx_left', 'by_left',
            'bx_right', 'by_right', 'box_size_left', 'box_size_right', 'disparity',
            'estimated_distance', 'est_dir_x', 'est_dir_y', 'est_dir_z',
            'real_dir_X', 'real_dir_Y', 'real_dir_Z'],
        '''
        log_data = self.log_data
        left_dir = self.left_dir
        right_dir= self.right_dir

        detection_left = self.detection_left
        detection_right = self.detection_right
        #[float], [float], [[float,float]]
        bx_left, by_left, left_boxes_size, left_center_list = self.detect_dict_to_center_list(detection_left)
        bx_right, by_right, right_boxes_size, right_center_list = self.detect_dict_to_center_list(detection_right)
        #[array([[...],[...]...]),array()...]
        disparity_list = self.get_disparity(log_data, left_dir,right_dir)
        #[float]
        pixel_values = self.get_pixel_values(disparity_list,left_center_list)

        

        player_pos = log_data[['Player_Pos_X', 'Player_Pos_Y', 'Player_Pos_Z']]
        enemy_pos = log_data[['Enemy_Pos_X', 'Enemy_Pos_Y', 'Enemy_Pos_Z']]
        log_data['bx_left'] = bx_left
        log_data['by_left'] = by_left
        log_data['bx_right'] = bx_right
        log_data['by_right'] = by_right
        log_data['box_size_left'] = left_boxes_size
        log_data['box_size_right'] = right_boxes_size
        log_data['disparity'] = pixel_values

                
        
        calc=DistCalculator()
        results = []
        for _, row in log_data.iterrows():
            left_pos = [row['StereoL_X'], row['StereoL_Y'], row['StereoL_Z']]
            left_rot = [row['StereoL_Roll'], row['StereoL_Pitch'], row['StereoL_Yaw']]
            right_pos = [row['StereoR_X'], row['StereoR_Y'], row['StereoR_Z']]
            right_rot = [row['StereoR_Roll'], row['StereoR_Pitch'], row['StereoR_Yaw']]

            midpoint = calc.compute_3d_position(
                row['bx_left'], row['by_left'], row['bx_right'], row['by_right'],
                left_pos, left_rot, right_pos, right_rot
            )
            results.append(midpoint)
        # 계산 결과를 새로운 데이터프레임으로 생성
        estimated_position = pd.DataFrame(results, columns=['X', 'Y', 'Z'])
        estimated_dist, estimated_dir = self.estimated_dist_angles(player_pos, estimated_position)   

        log_data['estimated_distance'] = estimated_dist['estimated_distance']
        log_data['est_dir_x'] = estimated_dir['X']
        log_data['est_dir_y'] = estimated_dir['Y']
        log_data['est_dir_z'] = estimated_dir['Z']
        #log_data['Player_Speed'] = self.get_velocity(log_data)     
        
        player_pos = log_data[['Player_Pos_X', 'Player_Pos_Y', 'Player_Pos_Z']]
        enemy_pos = log_data[['Enemy_Pos_X', 'Enemy_Pos_Y', 'Enemy_Pos_Z']]
        log_data['real_dir_X']=(enemy_pos['Enemy_Pos_X']-player_pos['Player_Pos_X'])/log_data['Distance']
        log_data['real_dir_Y']=(enemy_pos['Enemy_Pos_Y']-player_pos['Player_Pos_Y'])/log_data['Distance']
        log_data['real_dir_Z']=(enemy_pos['Enemy_Pos_Z']-player_pos['Player_Pos_Z'])/log_data['Distance']
        
        #log_data = pd.concat([log_data,real_dir],axis=1)

        return log_data



class StereoRegression:
    def __init__(self):
        return

    def regression_model(self, log_data):
        '''
        return log_data (pd.DataFrame) :
            ['Time', 'Distance', 'Player_Pos_X', 'Player_Pos_Y', 'Player_Pos_Z',
            'Player_Speed', 'Player_Health', 'Player_Turret_X', 'Player_Turret_Y',
            'Player_Body_X', 'Player_Body_Y', 'Player_Body_Z', 'TurretCam_X',
            'TurretCam_Y', 'TurretCam_Z', 'StereoL_X', 'StereoL_Y', 'StereoL_Z',
            'StereoL_Roll', 'StereoL_Pitch', 'StereoL_Yaw', 'StereoR_X',
            'StereoR_Y', 'StereoR_Z', 'StereoR_Roll', 'StereoR_Pitch',
            'StereoR_Yaw', 'Enemy_Pos_X', 'Enemy_Pos_Y', 'Enemy_Pos_Z',
            'Enemy_Speed', 'Enemy_Health', 'Enemy_Turret_X', 'Enemy_Turret_Y',
            'Enemy_Body_X', 'Enemy_Body_Y', 'Enemy_Body_Z', 'bx_left', 'by_left',
            'bx_right', 'by_right', 'box_size_left', 'box_size_right', 'disparity',
            'estimated_distance', 'est_dir_x', 'est_dir_y', 'est_dir_z',
            'real_dir_X', 'real_dir_Y', 'real_dir_Z'],
        '''
        log_data=log_data 
        # 종속 변수(y)와 독립 변수(X) 분리
        y = log_data[['Distance', 'Enemy_Pos_X', 'Enemy_Pos_Y', 'Enemy_Pos_Z']]
        X = log_data[['Player_Pos_X', 'Player_Pos_Y', 'Player_Pos_Z','Player_Body_X', 'Player_Body_Y', 'Player_Body_Z',
                      'StereoL_X', 'StereoL_Y', 'StereoL_Z', 'StereoL_Roll', 'StereoL_Pitch', 'StereoL_Yaw', 'StereoR_X','StereoR_Y', 'StereoR_Z',
                      'bx_left', 'by_left','bx_right', 'by_right', 'box_size_left', 'box_size_right',  
                      'estimated_distance','disparity']] #'est_dir_x', 'est_dir_y', 'est_dir_z'
        

        # 학습/테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 모델 정의 및 학습 (Random Forest Regressor 사용)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 모델 저장
        joblib.dump(model, 'random_forest_model.pkl')
        
        # 예측
        y_pred = model.predict(X_test)

        # 성능 평가
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"R² Score: {r2:.3f}")

        return model, mse, r2, y_test, y_pred

"""
사용 예시
from StereoRegression import StereoPreprocess, StereoRegression
left_dir = "C:/Users/Dhan/Documents/Tank Challenge/capture_images/L"
right_dir = "C:/Users/Dhan/Documents/Tank Challenge/capture_images/R"
log_path = "C:/Users/Dhan/Documents/Tank Challenge/log_data/tank_info_log.txt"
Preprocess = StereoPreprocess(left_dir, right_dir, log_path)
log_data = Preprocess.get_log_data()
Reg = StereoRegression()
model, mse, rw = Reg.regression_model(log_data)
"""