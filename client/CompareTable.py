import os, torch, cv2, glob
import pandas as pd
import numpy as np
from ultralytics import YOLO
from modules.StereoRegression import DistCalculator,StereoPreprocess, StereoRegression
import joblib


class CompareTable:
    def __init__(self, model_path=None):
        if model_path is None:
            self.model_path = 'best.pt'
        else:
            self.model_path = model_path
        print(self.model_path)
        self.model = YOLO(self.model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_classes = {
            0: 'Car002', 1: 'Car003', 2: 'Car005', 3: 'Human001',
            4: 'Rock001', 5: 'Rock2', 6: 'Tank001', 7: 'Wall001', 8: 'Wall002'
        }
        return
    def detect(self, image_path):
        """
        image_path에 해당하는 이미지 하나에 대한 이미지 디텍션
        """
        results = self.model(image_path)
        detections = results[0].boxes.data.cpu().numpy()
        
        filtered_results = []
        for box in detections:
            class_id = int(box[5])
            if class_id == 6:  # tank
                filtered_results.append({
                    'className': self.target_classes[class_id],
                    'bbox': [float(coord) for coord in box[:4]],
                    'confidence': float(box[4])
                })
        return filtered_results
    
    def get_obj_center(self, img_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO('best.pt').to(device)
        results = model(img_path)
        detections = results[0].boxes.data.cpu().numpy()
    
        if len(detections) == 0:
            raise ValueError("객체가 탐지되지 않았습니다.")

        # 예: 첫 번째 탐지 객체만 사용
        box = detections[0]
        bx = int((box[0] + box[2]) / 2)
        by = int((box[1] + box[3]) / 2)
        return bx, by

    # =========================
    # [2] 스테레오 거리 추정 파이프라인
    # =========================
    def stereo_distance_pipeline(self, left_img_path, right_img_path, Q):
        # (1) 이미지 불러오기
        imgL = cv2.imread(left_img_path, 0)
        imgR = cv2.imread(right_img_path, 0)

        # (2) 스테레오 정합 설정
        stereo = cv2.StereoSGBM_create(
            numDisparities=16 * 5,
            blockSize=9,
            minDisparity=0,
            P1=8 * 3 * 3 ** 2,
            P2=32 * 3 * 3 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

        # (3) Q 행렬 기반으로 3D 포인트 재구성
        points_3D = cv2.reprojectImageTo3D(disparity, Q)

        # (4) YOLO로 바운딩 박스 중심 추출 (왼쪽 이미지 기준)
        bx, by = self.get_obj_center(left_img_path)

        # (5) 해당 픽셀 위치의 3D 좌표 반환
        point_3D = points_3D[by, bx]
        return point_3D
    def detection2list(self,img_path,detect_model_path = None):
        if detect_model_path is None:
            detect_model_path = self.model_path
        detect_model = YOLO(detect_model_path)
        target_classes =self.target_classes
        results = detect_model(img_path)
        detections = results[0].boxes.data.cpu().numpy()
        detection_list = []
        if np.any(detections):
            for detection in detections:
                class_id = int(detection[5])
                if class_id in target_classes:
                    [x1,y1,x2,y2] = [float(coord) for coord in detection[:4]]
                    box_size = (x2-x1)*(y2-y1)
                    bx = (x2+x1)/2
                    by = (y2+y1)/2
                    detection_list.append(
                        {   
                            'className' : target_classes[class_id],
                            'bbox' : [x1,y1,x2,y2],
                            'box_center': [bx,by],
                            'box_size' : box_size                
                    }
                ) 
        return detection_list



    def match_detections(self,detections_left, detections_right, y_thresh=20, disparity_thresh=200):
        matches = []
        used_right = set()

        for det_L in detections_left:
            best_match = None
            best_score = float('inf')

            x_center_L, y_center_L = det_L['box_center']

            for idx, det_R in enumerate(detections_right):
                if idx in used_right:
                    continue
                if det_L['className'] != det_R['className']:
                    continue

                x_center_R, y_center_R = det_R['box_center']

                # 조건 1: y 중심 유사
                if abs(y_center_L - y_center_R) > y_thresh:
                    continue

                # 조건 2: 디스패리티
                disparity = x_center_L - x_center_R
                if disparity <= 0 or disparity > disparity_thresh:
                    continue

                score = abs(y_center_L - y_center_R) + abs(disparity)
                if score < best_score:
                    best_score = score
                    best_match = idx

            if best_match is not None:
                matches.append([det_L, detections_right[best_match]])
                used_right.add(best_match)

        # ✅ 중심 x좌표 기준 정렬 (왼쪽 객체 기준)
        matches_sorted = sorted(
            matches,
            key=lambda pair: pair[0]['box_center'][0]
        )

        return matches_sorted

    def add_estimated_dist_to_matches(self,matches,latest_log):
        calc = DistCalculator()
        stereoL_pos = latest_log[['StereoL_X', 'StereoL_Y', 'StereoL_Z']].squeeze()
        stereoL_rot =latest_log[['StereoL_Roll', 'StereoL_Pitch', 'StereoL_Yaw']].squeeze()
        stereoR_pos = latest_log[['StereoR_X','StereoR_Y', 'StereoR_Z']].squeeze()
        stereoR_rot = latest_log[['StereoR_Roll', 'StereoR_Pitch', 'StereoR_Yaw']].squeeze()
        for idx, detection_pair in enumerate(matches):
            det_L = detection_pair[0]
            bx_left, by_left = det_L['box_center']
            det_R = detection_pair[1]
            bx_right, by_right = det_R['box_center']
            estimated_coord = calc.compute_3d_position(bx_left,by_left,bx_right,by_right,stereoL_pos, stereoL_rot, stereoR_pos, stereoR_rot)
            matches[idx].append(estimated_coord)
    
        return matches #[[{detection_left},{detection_right},[estimated_coord]]]

    def img2disparity(self,left_img_path, right_img_path):
        img_L = cv2.imread(left_img_path,cv2.IMREAD_GRAYSCALE)
        img_R = cv2.imread(right_img_path,cv2.IMREAD_GRAYSCALE)
        stereo = cv2.StereoSGBM_create(minDisparity=0,
                                    numDisparities=64,
                                blockSize=9,
                                P1=8 * 3 * 9 ** 2,
                                P2=32 * 3 * 9 ** 2,
                                disp12MaxDiff=1,
                                uniquenessRatio=10,
                                speckleWindowSize=100,
                                speckleRange=32)
        disparity_img = stereo.compute(img_L, img_R).astype(np.float32) / 16.0
        return disparity_img

    def log2pred(self, left_img_path, right_img_path, log_row):
        latest_log=log_row
        left_detection_list=self.detection2list(left_img_path)
        right_detection_list=self.detection2list(right_img_path)
        
        detection_matched = self.match_detections(left_detection_list,right_detection_list)
    
        disparity_img = self.img2disparity(left_img_path,right_img_path)

        #반복문 돌리는 기준
        detection_with_coord = self.add_estimated_dist_to_matches(detection_matched,latest_log)
        log_X = latest_log[['Player_Pos_X', 'Player_Pos_Y', 'Player_Pos_Z','Player_Body_X', 'Player_Body_Y', 'Player_Body_Z',
                            'StereoL_X', 'StereoL_Y', 'StereoL_Z', 'StereoL_Roll', 'StereoL_Pitch', 'StereoL_Yaw', 'StereoR_X','StereoR_Y', 'StereoR_Z']]
        player_pos = latest_log[['Player_Pos_X', 'Player_Pos_Y', 'Player_Pos_Z']]
        y_pred_list=[]
        triangulation_result = {}
        for idx, detection_info in enumerate(detection_with_coord):
            className = detection_info[0]['className']
            bx_left, by_left = detection_info[0]['box_center']
            bx_right, by_right =  detection_info[1]['box_center'] 
            box_size_left = detection_info[0]['box_size']
            box_size_right = detection_info[1]['box_size']
            estimated_dist = np.linalg.norm(detection_info[2]-player_pos)

            triangulation_result['x'] = detection_info[2][0]
            triangulation_result['z'] = detection_info[2][2]
            triangulation_result['y'] = detection_info[2][1]
            triangulation_result['distance'] = estimated_dist

            disparity_val = disparity_img[int(round(by_left,0))][int(round(bx_left,0))]
            if disparity_val<0:
                disparity_val=0
            else:
                disparity_val=disparity_val

            box_info_dict={'bx_left': bx_left, 'by_left': by_left,
                           'bx_right': bx_right, 'by_right': by_right,
                           'box_size_left': box_size_left, 'box_size_right': box_size_right,
                           'estimated_distance': estimated_dist,
                           'disparity': disparity_val}
            log_dict=log_X.to_dict(orient='records')[0]
            combined_dict = {**log_dict, **box_info_dict}
            X = pd.DataFrame([combined_dict])
            inference_model = joblib.load('random_forest_model.pkl')
            y_pred = inference_model.predict(X)
            y_pred = np.append(className,y_pred) # (ndarray)[className,Distance,X,Y,Z]
            #========================= y_pred 딕셔너리로 바꿔주기 =================================
            keys = ['x', 'z', 'y', 'className', 'distance', 'id']
            values = y_pred.tolist()
            id = f'{values[0]}_{idx}'
            y_pred_dict={keys[0]:values[2], keys[1]:values[4], keys[2]:values[3], keys[3]:values[0], keys[4]:values[1], keys[5]:id} 
            y_pred_list.append(y_pred_dict)

        return {'list':y_pred_list,'state':True}, triangulation_result #(array)[  [className,Distance,X,Y,Z],    [...],[...],...]
    
    def compare_table(self, left_img_dir, right_img_dir, log_path):
        log_data = pd.read_csv(log_path)
        image_files = glob.glob(os.path.join(left_img_dir, '*.png'))
        fx = 889.17
        fy =889.17
        cx = 512
        cy = 512
        baseline = 1.0

        Q = np.float32([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, fx],
            [0, 0, -1/baseline, 0]
            ])
        
        reg_tri_results=[]
        for filename in image_files:
            basename = os.path.splitext(os.path.basename(filename))[0]

            log_matched = log_data[log_data['Time'] == float(basename)]

            results = []
            if not log_matched.empty:
                left_img_path = os.path.join(left_img_dir, basename + '.png')
                right_img_path = os.path.join(right_img_dir, basename + '.png')
                reg_pred, tri_pred = self.log2pred(left_img_path, right_img_path, log_matched)
                #disp_pred = self.stereo_distance_pipeline(left_img_path, right_img_path, Q)
                for reg_result in reg_pred['list']:
                    result = {'Time':basename,
                              #'obj_pos':log_matched[['']], #obj distance - pred_distance < threshhold 
                              'reg_pred':reg_result,
                              'tri_pred': tri_pred}
                    results.append(result)
            else:
                print(f"No log match for {basename}")
            reg_tri_results.append(results)        

        return reg_tri_results

    
left_dir = 'C:\\Users\\Dhan\\Desktop\\Project3\\snapshot\\L'
right_dir = 'C:\\Users\\Dhan\\Desktop\\Project3\\snapshot\\R'
log_path = 'C:\\Users\\Dhan\\Desktop\\Project3\\snapshot\\tank_info_log.txt'
table_maker = CompareTable(model_path='C:\\Users\\Dhan\\Desktop\\Project3\\best.pt')
results = table_maker.compare_table(left_dir, right_dir, log_path)

print(results)
