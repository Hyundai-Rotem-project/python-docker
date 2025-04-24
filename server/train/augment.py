import os
import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from tqdm import tqdm
from typing import List, Tuple, Any
import gc
import random

# 이미지, 라벨벨 읽기 및 저장
class ImageFileManager:
    def __init__(self, image_dir, label_dir, output_image_dir, output_label_dir, batch_size=100):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_image_dir = output_image_dir
        self.output_label_dir = output_label_dir
        self.batch_size = batch_size
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

    def __read_labels__(self, label_path):
        bboxes = []
        class_labels = []

        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = parts[0]
                bbox = [float(i) for i in parts[1:]]
                
                bboxes.append(bbox)
                class_labels.append(int(float(class_id)))
        return bboxes, class_labels

    def __get_image_batches__(self):
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        total_images = len(image_files)
        
        print(f"\n이미지 로딩 중... 총 {total_images}개 파일")
        print(f"배치 크기: {self.batch_size}")
        
        for i in range(0, total_images, self.batch_size):
            batch_files = image_files[i:i + self.batch_size]
            batch_data = []
            
            for filename in tqdm(batch_files, desc=f"배치 처리 중 ({i+1}-{min(i+self.batch_size, total_images)}/{total_images})"):
                image_path = os.path.join(self.image_dir, filename)
                label_path = os.path.join(self.label_dir, filename.replace('.jpg', '.txt'))
                
                image = cv2.imread(image_path)
                if image is None:
                    continue

                bboxes, class_labels = self.__read_labels__(label_path)
                batch_data.append((filename, image, bboxes, class_labels))
            
            yield batch_data
            
            del batch_data
            gc.collect()
    
    def __save_output__(self, filename, data, index):
        image, bboxes, labels = data

        name, _ = os.path.splitext(filename)
        output_images_path = os.path.join(self.output_image_dir, f'{name}_{index}.jpg')
        output_labels_path = os.path.join(self.output_label_dir, f'{name}_{index}.txt')

        cv2.imwrite(output_images_path, image)
        with open(output_labels_path, 'w') as f:
            for label, bbox in zip(labels, bboxes):
                x, y, w, h = bbox 
                f.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# 이미지 객체 합성
class ObjectSynthesizer(ImageFileManager):
    def __init__(self, image_dir, label_dir, output_image_dir, output_label_dir, obj_num, batch_size=100):
        super().__init__(image_dir, label_dir, output_image_dir, output_label_dir, batch_size)
        self.obj_num = obj_num

    def __get_random_image__(self, batch_data):
        idx_list = np.random.choice(len(batch_data), size=self.obj_num, replace=False)
        random_images = list(map(lambda x: batch_data[x], idx_list))[0]
        return random_images[1:]

    def __get_obj_info__(self, img, bbox):
        xc, yc, w, h = bbox
        ih, iw = img.shape[:2]
        x1 = int((xc - w/2) * iw)
        x2 = int((xc + w/2) * iw)
        y1 = int((yc - h/2) * ih)
        y2 = int((yc + h/2) * ih)

        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid bbox range: {bbox}")
        
        obj = img[y1:y2, x1:x2] 
        obj_w = x2 - x1
        obj_h = y2 - y1

        if obj_w == 0 or obj_h == 0:
            raise ValueError("Object width/height is zero after cropping.")
        
        return (obj, obj_w, obj_h)

    def run(self):
        print("\n객체 합성 시작...")
        for batch_data in self.__get_image_batches__():
            for filename, image, bboxes, class_labels in tqdm(batch_data, desc="객체 합성"):
                try:
                    target_h, target_w = image.shape[:2]
                    new_img = image.copy()
                    new_bboxes = list(bboxes)
                    new_labels = list(class_labels)
                    
                    obj_img, obj_bboxes, obj_labels = self.__get_random_image__(batch_data)

                    for index, (bbox, label) in enumerate(zip(obj_bboxes, obj_labels)):
                        obj, obj_w, obj_h = self.__get_obj_info__(obj_img, bbox)
                        max_x = target_w - obj_w
                        max_y = target_h - obj_h
                        
                        if max_x <= 0 or max_y <= 0:
                            print(f"Object too big for target image, skipping...")
                            continue

                        new_x = np.random.randint(0, max_x)
                        new_y = np.random.randint(0, max_y)
                        
                        new_img[new_y:new_y+obj_h, new_x:new_x+obj_w] = obj
                        new_bbox = self.__get_yolo_data__(new_x, obj_w, target_w, new_y, obj_h, target_h)
                        
                        new_bboxes.append(new_bbox)
                        new_labels.append(label)
                        
                        data = (new_img, new_bboxes, new_labels)
                        self.__save_output__(filename, data, index)
                except Exception as e:
                    print(f"객체 합성 중 오류 발생: {str(e)}")
                    continue

# 이미지 증강 및 저장
class Augmentator(ImageFileManager):
    def __init__(self, image_dir, label_dir, output_image_dir, output_label_dir, transform, output_num=1, batch_size=100, day_or_night="both"):
        super().__init__(image_dir, label_dir, output_image_dir, output_label_dir, batch_size)
        self.transform = transform
        self.output_num = output_num
        self.day_or_night = day_or_night.lower()
        
        # 효과 분리
        self.night_effects = [
            self.add_wind,
            self.add_thunderstorm,
            self.add_tornado,
            self.add_falling_leaves
        ]

        self.day_effects = [
            self.add_wind,
            self.add_rainbow,
            self.add_tornado,
            self.add_falling_leaves
        ]

    def add_wind(self, img):
        blurred = cv2.GaussianBlur(img, (11, 3), sigmaX=10)
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        windy = cv2.add(blurred, noise)
        return windy
    
    def add_thunderstorm(self, img):
        img = cv2.convertScaleAbs(img, alpha=0.5, beta=-30)
        mask = np.zeros_like(img, dtype=np.uint8)
        for _ in range(2):
            x, y = np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])
            radius = np.random.randint(50, 120)
            cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)
        lightning = cv2.addWeighted(img, 1.0, mask, 0.6, 0)
        return lightning
    
    def add_rainbow(self, img):
        rainbow = np.zeros_like(img)
        h, w = img.shape[:2]
        center = (w // 2, h)
        for i, color in enumerate([(255, 0, 0), (255, 127, 0), (255, 255, 0), 
                                (0, 255, 0), (0, 0, 255), (75, 0, 130), (143, 0, 255)]):
            cv2.ellipse(rainbow, center, (int(w*0.8)-i*8, int(h*0.5)-i*8), 0, 0, 180, color, thickness=6)
        rainbowed = cv2.addWeighted(img, 1.0, rainbow, 0.3, 0)
        return rainbowed

    def add_night(self, img):
        dark = cv2.convertScaleAbs(img, alpha=0.5, beta=-50)
        night = cv2.addWeighted(dark, 1.0, np.full_like(img, (10, 10, 30)), 0.3, 0)
        return night
    
    def add_tornado(self, img):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        strength = 0.0008

        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                dx = x - center[0]
                dy = y - center[1]
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx) + strength * r
                map_x[y, x] = center[0] + r * np.cos(theta)
                map_y[y, x] = center[1] + r * np.sin(theta)

        swirl = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        swirl_blur = cv2.GaussianBlur(swirl, (5, 5), sigmaX=3)
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        tornado = cv2.add(swirl_blur, noise)
        
        return tornado
    
    def add_falling_leaves(self, img, num_leaves=30):
        h, w = img.shape[:2]
        leaf_img = np.zeros_like(img)

        for _ in range(num_leaves):
            color = random.choice([(165, 42, 42), (255, 140, 0), (255, 215, 0), (139, 69, 19)])
            center = (random.randint(0, w), random.randint(0, h))
            size = random.randint(10, 30)
            angle = random.uniform(0, 360)
            
            axes = (size, int(size * 0.4))
            temp = np.zeros_like(img)
            cv2.ellipse(temp, center, axes, angle, 0, 360, color, -1)

            alpha = random.uniform(0.3, 0.7)
            leaf_img = cv2.addWeighted(leaf_img, 1.0, temp, alpha, 0)

        result = cv2.addWeighted(img, 1.0, leaf_img, 0.4, 0)
        return result

    def run(self):
        print("\n이미지 증강 시작...")
        for batch_data in self.__get_image_batches__():
            for filename, image, bboxes, class_labels in tqdm(batch_data, desc="이미지 증강"):
                try:
                    for index in range(self.output_num):
                        # Albumentation 증강
                        augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                        aug_image = augmented['image']
                        
                        # OpenCV 증강: 1~3개 무작위 조합
                        if self.day_or_night == "day":
                            available_effects = self.day_effects
                        elif self.day_or_night == "night":
                            available_effects = self.night_effects
                        else:  # "both"일 경우 랜덤 선택
                            available_effects = random.choice([self.day_effects, self.night_effects])

                        num_effects = random.randint(1, min(3, len(available_effects)))
                        selected_effects = random.sample(available_effects, num_effects)

                        for effect in selected_effects:
                            aug_image = effect(aug_image)
                        
                        data = (aug_image, augmented['bboxes'], augmented['class_labels'])
                        self.__save_output__(filename, data, index)
                except Exception as e:
                    print(f"이미지 증강 중 오류 발생: {str(e)}")
                    continue

# 이미지 증강 시퀀스 예제
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.3
        ),
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.Blur(blur_limit=(3, 5), p=0.5),
        ], p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.2),
        A.OneOf([
            # 안개
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.6, alpha_coef=0.1, p=1.0),

            # 비
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                blur_value=3,
                brightness_coefficient=0.7,
                rain_type=None,
                p=1.0
            ),

            # 눈
            A.RandomSnow(
                snow_point_lower=0.1,
                snow_point_upper=0.3,
                brightness_coeff=1.2,
                p=1.0
            ),

            # 햇빛/플레어
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),  # 상단 50% 영역에만 적용
                angle_lower=0.3,
                src_radius=100,
                src_color=(255, 255, 255),
                p=1.0
            ),

            # 그림자 (나무, 건물 등)
            A.RandomShadow(
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                shadow_roi=(0, 0.5, 1, 1),  # 하단 절반에만 그림자
                p=1.0
            ),
        ], p=0.3),
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
) 

if __name__ == "__main__":
    # 경로 설정
    image_dir = '/data/roboflow_image_dataset/train/images/'
    label_dir = '/data/roboflow_image_dataset/train/labels/'
    output_image_dir = '/data/roboflow_image_dataset/train/aug_images/'
    output_label_dir = '/data/roboflow_image_dataset/train/aug_labels/'

    # 배치 크기 설정
    BATCH_SIZE = 100

    print("\n=== 이미지 증강 및 합성 시작 ===")
    print(f"입력 이미지 경로: {image_dir}")
    print(f"입력 라벨 경로: {label_dir}")
    print(f"출력 이미지 경로: {output_image_dir}")
    print(f"출력 라벨 경로: {output_label_dir}")

    try:
        # 이미지 증강 실행
        aug = Augmentator(
            image_dir=image_dir,
            label_dir=label_dir,
            output_image_dir=output_image_dir,
            output_label_dir=output_label_dir,
            transform=transform,
            output_num=1,
            batch_size=BATCH_SIZE,
            day_or_night="both"  # "day", "night", "both" 중 선택
        )
        aug.run()

        # 객체 합성 실행
        synth = ObjectSynthesizer(
            image_dir=image_dir,
            label_dir=label_dir,
            output_image_dir=output_image_dir,
            output_label_dir=output_label_dir,
            obj_num=3,
            batch_size=BATCH_SIZE
        )
        synth.run()

        print("\n=== 이미지 증강 및 합성 완료 ===")
        print(f"결과물 확인: {output_image_dir}")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        print("프로그램이 비정상적으로 종료되었습니다.")