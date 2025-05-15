import os, torch, cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from StereoRegression import DistCalculator,StereoPreprocess, StereoRegression
import joblib

left_dir = "C:/Users/Dhan/Documents/Tank Challenge/capture_images/L"
right_dir = "C:/Users/Dhan/Documents/Tank Challenge/capture_images/R"
log_path = "C:/Users/Dhan/Documents/Tank Challenge/log_data/tank_info_log.txt"
detect_model_path = 'best.pt'
detect_model=YOLO(detect_model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
target_classes = {
            0: 'car2', 1: 'car3', 2: 'car5', 3: 'human', 4: 'rock',
            5: 'tank', 6: 'wall'
        }
class RealTimeInference:
    def __init__(self, model_path=None):
        if model_path is None:
            self.model_path = 'best.pt'
        self.model = YOLO(self.model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_classes = {
            0: 'car2', 1: 'car3', 2: 'car5', 3: 'human', 4: 'rock',
            5: 'tank', 6: 'wall'
        }
    def train_model(left_dir, right_dir, log_path):
        Preprocess = StereoPreprocess(left_dir, right_dir, log_path)
        log_data = Preprocess.get_log_data()
        Reg = StereoRegression()
        inference_model, mse, r2, y_test, y_pred = Reg.regression_model(log_data)
        return inference_model
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

    def prepare_log(self,log_path):
        """
        return : 
            y : ['Distance', 'Enemy_Pos_X', 'Enemy_Pos_Y', 'Enemy_Pos_Z']

            X : ['Player_Pos_X', 'Player_Pos_Y', 'Player_Pos_Z','Player_Body_X', 'Player_Body_Y', 'Player_Body_Z',
                'StereoL_X', 'StereoL_Y', 'StereoL_Z', 'StereoL_Roll', 'StereoL_Pitch', 'StereoL_Yaw', 'StereoR_X','StereoR_Y', 'StereoR_Z',
                'bx_left', 'by_left','bx_right', 'by_right', 'box_size_left', 'box_size_right', 'disparity', 
                'estimated_distance', 'est_dir_x', 'est_dir_y', 'est_dir_z']
        """
        # 가장 마지막 로그
        log_data=pd.read_csv(log_path)
        latest_log = log_data.iloc[-1]
    
        return latest_log
    def preapare_image(self,left_dir,right_dir):
        # 폴더 내 모든 파일 목록 가져오기
        left_files = os.listdir(left_dir)
        right_files = os.listdir(right_dir)
        # 마지막 이미지 선택
        sorted_files = sorted(left_files, key=lambda x: float(x.replace('.png', '')))
        left_img_path = os.path.join(left_dir, sorted_files[-1])
        right_img_path = os.path.join(right_dir, sorted_files[-1])

        return left_img_path, right_img_path

    def detection2list(self,img_path,detect_model_path = None):
        if detect_model_path is None:
            detect_model_path = 'best.pt'
        detect_model=YOLO(detect_model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
        target_classes = {
                0: 'car2', 1: 'car3', 2: 'car5', 3: 'human', 4: 'rock',
                5: 'tank', 6: 'wall'
            }
        results = detect_model(img_path)
        detections = results[0].boxes.data.cpu().numpy()
        detection_list = []
        for detection in detections:
            class_id = detection[5]
            if class_id in target_classes:
                [x1,y1,x2,y2] = [float(coord) for coord in detection[:4]]
            box_size = (x2-x1)*(y2-y1)
            bx = (x2+x1)/2
            by = (y2+y1)/2
            detection_list.append(
                {   
                    'obj_name' : target_classes[class_id],
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
                if det_L['obj_name'] != det_R['obj_name']:
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
        stereoL_pos = latest_log[['StereoL_X', 'StereoL_Y', 'StereoL_Z']] 
        stereoL_rot =latest_log[['StereoL_Roll', 'StereoL_Pitch', 'StereoL_Yaw']]
        stereoR_pos = latest_log[['StereoR_X','StereoR_Y', 'StereoR_Z']]
        stereoR_rot = latest_log[['StereoR_Roll', 'StereoR_Pitch', 'StereoR_Yaw']]
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

    def get_disparity():
        return

    def log2pred(self, left_dir, right_dir, log_path):
        latest_log=self.prepare_log(log_path)
        left_img_path, right_img_path = self.preapare_image(left_dir,right_dir)
        left_detection_list=self.detection2list(left_img_path)
        right_detection_list=self.detection2list(right_img_path)
        
        detection_matched = self.match_detections(left_detection_list,right_detection_list)
    
        disparity_img = self.img2disparity(left_img_path,right_img_path)

        #반복문 돌리는 기준
        detection_with_coord = self.add_estimated_dist_to_matches(detection_matched,latest_log)
        log_X = latest_log[['Player_Pos_X', 'Player_Pos_Y', 'Player_Pos_Z','Player_Body_X', 'Player_Body_Y', 'Player_Body_Z',
                            'StereoL_X', 'StereoL_Y', 'StereoL_Z', 'StereoL_Roll', 'StereoL_Pitch', 'StereoL_Yaw', 'StereoR_X','StereoR_Y', 'StereoR_Z']]
        y_pred_list=[]
        for detection_info in detection_with_coord:
            bx_left, by_left = detection_info[0]['box_center']
            bx_right, by_right =  detection_info[1]['box_center'] 
            box_size_left = detection_info[0]['box_size']
            box_size_right = detection_info[1]['box_size']
            estimated_dist = np.linalg.norm(detection_info[2])
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
            log_dict=log_X.to_dict()
            combined_dict = {**log_dict, **box_info_dict}
            X = pd.DataFrame([combined_dict])
            inference_model = joblib.load('random_forest_model.pkl')
            y_pred = inference_model.predict(X)
            y_pred_list.append(y_pred)

        return y_pred_list

inference = RealTimeInference()
y_pred = inference.log2pred(left_dir,right_dir,log_path)
log = inference.prepare_log(log_path)
print('y_pred:',y_pred)
print('log',log[['Enemy_Pos_X','Enemy_Pos_Y','Enemy_Pos_Z']])


