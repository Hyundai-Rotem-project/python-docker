import os
import glob
import pandas as pd
import torch
from ultralytics import YOLO

class StereoImageFilter:
    def __init__(self, left_dir, right_dir, log_path=None, model_path=None):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.log_path = log_path
        self.log_data = pd.read_csv(self.log_path)
        self.model_path = model_path
        if model_path is None:
            self.model_path = 'C:\\Users\\Dhan\\Desktop\\Project3\\best.pt'
        self.model = YOLO(self.model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_classes = {
            0: 'car2', 1: 'car3', 2: 'car5', 3: 'human1', 4: 'rock1',
            5: 'rock2', 6: 'tank', 7: 'wall1', 8: 'wall2'
        }


    def filename_log_match(self, image_dir):
        """
        image_dir에 있는 stereo 이미지 이름을 log_data의 최근 시간으로 변경
        """
        if self.log_path is None:
            print("⚠️ 로그 파일 경로가 지정되지 않았습니다.")
            return

        log_data = self.log_data
        image_files = glob.glob(os.path.join(image_dir, '*.png'))

        image_time_map = {}
        for file_path in image_files:
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            num_str = base_name.replace('_', '')
            try:
                img_time_int = int(num_str)
                image_time_map[img_time_int] = file_path
            except ValueError:
                continue

        if not image_time_map:
            print("⚠️ 이미지에서 시간 추출 실패 또는 파일 없음")
            return

        used_image_paths = set()

        for time in log_data['Time']:
            time_int = int(round(time * 100))
            closest_img_time = min(image_time_map.keys(), key=lambda x: abs(x - time_int))
            src_path = image_time_map[closest_img_time]
            new_filename = f"{time:.2f}.png"
            dst_path = os.path.join(image_dir, new_filename)

            if not os.path.exists(dst_path):
                os.rename(src_path, dst_path)
                used_image_paths.add(dst_path)
            else:
                print(f"⚠️ {dst_path} 이미 존재하여 생략")

            del image_time_map[closest_img_time]

        all_images_after = glob.glob(os.path.join(image_dir, '*.png'))
        for img_path in all_images_after:
            if img_path not in used_image_paths:
                os.remove(img_path)
        
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


    def filter_image(self, image_dir):
        """
        객체가 탐지되지 않은 이미지를 삭제하고, 해당 이미지에 대응되는 log_data의 Time 행도 삭제
        
        return : 
            filtered_log_data (list of dictionary)
            detect_result (dictionary) : { time:[ {'className':'object','bbox':'x1,y1,x2,y2',confidence:float} ,...],... }

        DataFrame filtered_log_data : 객체가 탐지 된 이미지에 대한 log data
        dict detect_result : 탐지 시 이미지 디텍션 결과 딕셔너리
        """
        log_data = self.log_data
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        filtered_log_data = pd.DataFrame(columns=log_data.columns)
        detect_results={}

        for image_path in image_files:
            detections = self.detect(image_path)

            # 이미지 이름에서 시간 추출 (예: '123.45.png' → 123.45)
            base = os.path.basename(image_path)
            image_name = float(os.path.splitext(base)[0])
            dict2update = {image_name:detections}
            detect_results.update(dict2update)

            if detections:
                # 시간 기준으로 log_data에서 해당 row 추출
                matched_row = log_data[log_data['Time'].round(2) == round(image_name, 2)]
                filtered_log_data = pd.concat([filtered_log_data, matched_row], ignore_index=True)
            else:
                os.remove(image_path)  # 탱크가 없는 이미지는 삭제

        return filtered_log_data, detect_results

    
    def filter_result(self, image_dir):
        """
        image_dir에 있는 이미지에 대하여 객체가 탐지된 이미지 필터링
        """
        self.filename_log_match(image_dir)
        filtered_log_data, detect_result = self.filter_image(image_dir)
        filtered_log_data = filtered_log_data.sort_values(by='Time').reset_index(drop=True)
        return filtered_log_data, detect_result
     
    def not_in_delete(self, image_dir, log_data):
        """
        필터링된 log 기준으로 객체가 탐지되지 않은 이미지 image_dir에서 삭제
        """
        files = glob.glob(os.path.join(image_dir, '*.png'))

        # Time 열을 소수점 둘째 자리로 반올림한 집합으로 변환
        log_time_set = set(log_data['Time'].round(2).tolist())

        for file_path in files:
            file_name = os.path.basename(file_path)
            try:
                image_time = float(os.path.splitext(file_name)[0])
            except ValueError:
                os.remove(file_path)
                continue

            # 반올림 후 비교
            if round(image_time, 2) not in log_time_set:
                os.remove(file_path)

    def folder_compare_and_delete(self,left_dir,right_dir):
        """
        left_dir, right_dir 교집합만 남기고 이미지, log_data 삭제
        return :
            new_log_data (pd.DataFrame) 
            left_detect_result_filtered (dictionary) : { time:[ {'className':'object','bbox':'x1,y1,x2,y2',confidence:float} ,...],... }
            right_detect_result_filtered (dictionary) : "

        """
        left_filtered_log, left_detect_result = self.filter_result(left_dir)
        right_filtered_log, right_detect_result = self.filter_result(right_dir)
        new_log_data = pd.merge(left_filtered_log, right_filtered_log, on=self.log_data.columns.to_list(), how='inner')

        left_detect_result_filtered = {}
        right_detect_result_filtered = {}
        for key, value in left_detect_result.items():
            if key in right_detect_result:
                left_detect_result_filtered[key] = (value)
                right_detect_result_filtered[key] = right_detect_result[key]
        

        #교집합에 없는 이미지 삭제
        self.not_in_delete(left_dir, new_log_data)
        self.not_in_delete(right_dir, new_log_data)

        return new_log_data, left_detect_result_filtered, right_detect_result_filtered
    
    def get_result(self):
        """
        return :
            filtered_log_data (pd.DataFrame) 
            filtered_detect_result (Dictionary) :  key:time, value:[detected_list]
                    {time: [{'className': 'str', 'bbox': [x1, y1, x2, y2], 'confidence': float},{...}], key:value,}  
        """
        filtered_log_data, left_filtered_detect_result, right_filtered_detect_result = self.folder_compare_and_delete(self.left_dir,self.right_dir)
        return filtered_log_data, left_filtered_detect_result, right_filtered_detect_result



"""
#사용 예시
# 경로 설정
left_folder = "C:/Users/Dhan/Documents/Tank Challenge/capture_images/L"
right_folder = "C:/Users/Dhan/Documents/Tank Challenge/capture_images/R"
log_path = "C:/Users/Dhan/Documents/Tank Challenge/log_data/tank_info_log.txt"

# 인스턴스 생성
filter = StereoImageFilter(left_dir=left_folder, right_dir=right_folder, log_path=log_path)

# 필터링 실행 return pandas DataFrame
log_filtered, left_detection, right_detection = filter.get_result()

# 결과 출력
print(log_filtered)

"""