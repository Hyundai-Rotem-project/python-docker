import torch
print(torch.cuda.is_available())  # True 출력
print(torch.version.cuda)         # 11.8 출력
print(torch.backends.cudnn.version())  # 8907 (cuDNN 8.9.7이면)

#==================================================================================================================================
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import cv2
import pandas as pd
import os
import torch
from ultralytics import YOLO

def pixel_to_ray(bx, by, camera_pos, camera_rotation, camera_intrinsic=None):
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
    rotation = R.from_euler('xyz', camera_rotation)
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

def compute_3d_position(player_pos, stereo_left_pos, stereo_left_rot, stereo_right_pos, stereo_right_rot,
                       bx_left, by_left, bx_right, by_right, camera_intrinsic=None):
    """
    스테레오 이미지에서 객체의 3D 좌표를 계산합니다.
    
    Args:
        player_pos: 플레이어 위치 (x, y, z)
        stereo_left_pos: 왼쪽 카메라 위치 (x, y, z)
        stereo_left_rot: 왼쪽 카메라 회전 (roll, pitch, yaw)
        stereo_right_pos: 오른쪽 카메라 위치 (x, y, z)
        stereo_right_rot: 오른쪽 카메라 회전 (roll, pitch, yaw)
        bx_left, by_left: 왼쪽 이미지에서의 객체 중심 좌표
        bx_right, by_right: 오른쪽 이미지에서의 객체 중심 좌표
        camera_intrinsic: 카메라 내부 파라미터 (선택사항)
    
    Returns:
        3D 좌표 (x, y, z)
    """
    # 각 카메라에서 광선 계산
    origin_left, direction_left = pixel_to_ray(bx_left, by_left, stereo_left_pos, stereo_left_rot, camera_intrinsic)
    origin_right, direction_right = pixel_to_ray(bx_right, by_right, stereo_right_pos, stereo_right_rot, camera_intrinsic)
    
    # 두 광선의 최근접점 계산
    midpoint, distance = find_closest_point_between_rays(origin_left, direction_left, origin_right, direction_right)
    
    # 거리가 너무 크면 삼각측량 실패로 간주
    if distance > 1.0:  # 임계값은 상황에 맞게 조정
        print(f"Warning: 두 광선 사이의 거리가 너무 큽니다: {distance}")
    
    return midpoint

def detect(image_path):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('best.pt')#.to(device)
    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    target_classes = {0: 'car2', 1: 'car3', 2: 'car5', 3: 'human1', 4: 'rock1', 5: 'rock2', 6: 'tank', 7: 'wall1', 8: 'wall2'}
    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4])
            })
    return filtered_results

def main():
    """
    columns
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
    # 입력값 설정
    left_folder = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\capture_images\\L"
    right_folder = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\capture_images\\R"
    log_path = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\log_data\\tank_info_log.txt"
    left_files = sorted(os.listdir(left_folder))
    right_files = sorted(os.listdir(right_folder))
    left_img_path = os.path.join(left_folder, left_files[-1])
    right_img_path = os.path.join(right_folder, right_files[-1])

    log = pd.read_csv(log_path)
    latest_log = log.iloc[-1,:]


    player_pos = np.array([latest_log['Player_Pos_X'],latest_log['Player_Pos_Y'],latest_log['Player_Pos_Z']])  # 플레이어 위치 (x, y, z)
    
    # 스테레오 카메라 설정 (예시 값)
    stereo_left_pos = np.array([latest_log['StereoL_X'],latest_log['StereoL_Y'],latest_log['StereoL_Z']])  # 왼쪽 카메라 위치
    stereo_left_rot = np.array([latest_log['StereoL_Roll'],latest_log['StereoL_Pitch'],latest_log['StereoL_Yaw']])  # 왼쪽 카메라 회전
    
    stereo_right_pos = np.array([latest_log['StereoR_X'],latest_log['StereoR_Y'],latest_log['StereoR_Z']])  # 오른쪽 카메라 위치
    stereo_right_rot = np.array([latest_log['StereoR_Roll'],latest_log['StereoR_Pitch'],latest_log['StereoR_Yaw']])  # 오른쪽 카메라 회전
    
    # 객체 중심 좌표 
    left_detection = detect(left_img_path)
    right_detection = detect(right_img_path)
    print(left_detection)
    x1, y1, x2, y2 = left_detection[0]['bbox']
    bx_left, by_left = ((x1+x2)/2), ((y1+y2)/2)
    x1, y1, x2, y2 = right_detection[0]['bbox']
    bx_right, by_right = ((x1+x2)/2), ((y1+y2)/2)
   

   


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
    # 3D 위치 계산
    object_position = compute_3d_position(
        player_pos, 
        stereo_left_pos, stereo_left_rot, 
        stereo_right_pos, stereo_right_rot,
        bx_left, by_left, bx_right, by_right,
        camera_intrinsic=camera_intrinsic
    )
    real_position = np.array([
        latest_log['Enemy_Pos_X'],
        latest_log['Enemy_Pos_Y'],
        latest_log['Enemy_Pos_Z']
    ])
    print(f'실제 객체 위치 : {real_position}')
    print(f"객체 추정 위치: {object_position}")
    print(f"오차: {np.linalg.norm(real_position - object_position)}")

if __name__ == "__main__":
    main()

#==================================================================================================================================
"""
파일 이름 바꾸기
"""
import pandas as pd
import os, glob, torch
from ultralytics import YOLO

# 이미지 파일이 있는 디렉토리 경로
left_folder = "C:/Users/Dhan/Documents/Tank Challenge/capture_images/L"
right_folder = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\capture_images\\R"
log_path = "C:\\Users\\Dhan\\Documents\\Tank Challenge\\log_data\\tank_info_log.txt"
#log_data=pd.read_csv(log_path)

def filename_log_match(log_path, image_dir):
    log_data = pd.read_csv(log_path)

    # 모든 이미지 파일 이름 수집 (예: '000_00.png')
    image_files = glob.glob(os.path.join(image_dir, '*.png'))

    # 이미지 파일 시간 매핑 딕셔너리
    image_time_map = {}
    for file_path in image_files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]  # '000_00'
        num_str = base_name.replace('_', '')       # '00000'
        try:
            img_time_int = int(num_str)
            image_time_map[img_time_int] = file_path
        except ValueError:
            continue

    if not image_time_map:
        print("⚠️ 이미지에서 시간 추출 실패 또는 파일 없음")
        return

    # 사용된 이미지 추적용 (삭제 방지용)
    used_image_paths = set()

    for time in log_data['Time']:
        time_int = int(round(time * 100))

        # 가장 가까운 시간의 이미지 선택
        closest_img_time = min(image_time_map.keys(), key=lambda x: abs(x - time_int))
        src_path = image_time_map[closest_img_time]

        new_filename = f"{time:.2f}.png"
        dst_path = os.path.join(image_dir, new_filename)

        # 이미지 이름 변경 (충돌 방지)
        if not os.path.exists(dst_path):
            os.rename(src_path, dst_path)
            used_image_paths.add(dst_path)
        else:
            print(f"⚠️ {dst_path} 이미 존재하여 생략")

        # 사용한 것 제거 (재사용 방지)
        del image_time_map[closest_img_time]

    # 남은 (사용되지 않은) 이미지 삭제
    all_images_after = glob.glob(os.path.join(image_dir, '*.png'))
    for img_path in all_images_after:
        if img_path not in used_image_paths:
            os.remove(img_path)

def detect(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('best.pt').to(device)
    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    target_classes = {0: 'car2', 1: 'car3', 2: 'car5', 3: 'human1', 4: 'rock1', 5: 'rock2', 6: 'tank', 7: 'wall1', 8: 'wall2'}
    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id == 6:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4])
            })
    return filtered_results

def filter_image(image_dir):
    image_files = glob.glob(os.path.join(image_dir, '*.png'))
    detection_result = []
    for image in image_files:
        detection = detect(image)
        if detection[0]['className']=='tank':
            detection[0]
            detection_result.append()
        else:
            os.remove(image)



filename_log_match(log_path,left_folder)
filename_log_match(log_path,right_folder)
