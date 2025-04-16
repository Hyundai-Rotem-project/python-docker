import os
import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
import random

# 이미지, 라벨 읽기 및 저장
class ImageFileManager:
    def __init__(self, image_dir, label_dir, output_image_dir, output_label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_image_dir = output_image_dir
        self.output_label_dir = output_label_dir
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

    def __read_labels__(self, label_path):
        bboxes = []
        class_labels = []

        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = parts[0]
                bbox = [float(i) for i in parts[1:]] # [x_center, y_center, width, height]
                
                bboxes.append(bbox)
                class_labels.append(int(float(class_id)))
        return bboxes, class_labels

    def __get_image_data__(self):
        image_meta_list = []
        for filename in os.listdir(self.image_dir):
            if not filename.endswith('.jpg'):
                continue

            image_path = os.path.join(self.image_dir, filename)
            label_path = os.path.join(self.label_dir, filename.replace('.jpg', '.txt'))
            
            image = cv2.imread(image_path)
            if image is None:
                continue

            bboxes, class_labels = self.__read_labels__(label_path)
            image_meta_list.append((filename, image, bboxes, class_labels))
        return image_meta_list # [(filename1, img1, bboxes1, labels1), (filename2, img2, bboxes2, labels2)]
    
    def __save_output__(self, filename, data, index):
        image, bboxes, labels = data

        name, _ = os.path.splitext(filename) # 파일 이름만 추출, 확장자 제거
        output_images_path = os.path.join(self.output_image_dir, f'{name}_{index}.jpg')
        output_labels_path = os.path.join(self.output_label_dir, f'{name}_{index}.txt')

        cv2.imwrite(output_images_path, image)
        with open(output_labels_path, 'w') as f:
            for label, bbox in zip(labels, bboxes):
                x, y, w, h = bbox 
                f.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# 이미지 객체 합성
class ObjectSynthesizer(ImageFileManager):
    def __init__(self, image_dir, label_dir, output_image_dir, output_label_dir, obj_num):
        super().__init__(image_dir, label_dir, output_image_dir, output_label_dir)
        self.obj_num = obj_num # 기본 이미지에 추가할 객체 개수
    
    def __get_random_image__(self):
        obj_images = self.__get_image_data__()
        idx_list = np.random.choice(len(obj_images), size=self.obj_num, replace=False)
        random_images = list(map(lambda x : obj_images[x], idx_list))[0]

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
    
    def __get_yolo_data__(self, new_x, obj_w, target_w, new_y, obj_h, target_h):
        # x1, x2, y1, y2 -> bbox
        new_xc = (new_x + obj_w/2) / target_w
        new_yc = (new_y + obj_h/2) / target_h
        new_w = obj_w / target_w
        new_h = obj_h / target_h
        return (new_xc, new_yc, new_w, new_h)

    def run(self):
        data = []
        for filename, image, bboxes, class_labels in self.__get_image_data__():
            target_img = image
            target_bboxes = bboxes
            target_labels = class_labels

            target_h, target_w = target_img.shape[:2]

            new_img = target_img.copy()
            new_bboxes = list(target_bboxes)
            new_labels = list(target_labels)
            
            obj_img, obj_bboxes, obj_labels = self.__get_random_image__()

            for index, (bbox, label) in enumerate(zip(obj_bboxes, obj_labels)):
                # 랜덤으로 얻은 객체를 랜덤위치에 붙여넣기
                obj, obj_w, obj_h = self.__get_obj_info__(obj_img, bbox)

                max_x = target_w - obj_w
                max_y = target_h - obj_h

                if max_x <= 0 or max_y <= 0:
                    print(f"Object too big for target image, skipping...")
                    continue  # 객체 크기가 너무 커서 붙이기 불가

                new_x = np.random.randint(0, max_x)
                new_y = np.random.randint(0, max_y)

                new_img[new_y:new_y+obj_h, new_x:new_x+obj_w] = obj

                new_bbox = self.__get_yolo_data__(new_x, obj_w, target_w, new_y, obj_h, target_h)

                new_bboxes.append(new_bbox)
                new_labels.append(label)
                print("??", new_img, new_bboxes, new_labels)

                data = (new_img, new_bboxes, new_labels)
                self.__save_output__(filename, data, index)


# 이미지 증강 및 저장
class Augmentator(ImageFileManager):
    def __init__(self, image_dir, label_dir, output_image_dir, output_label_dir, transform, output_num=1,day_or_night="both"):
        super().__init__(image_dir, label_dir, output_image_dir, output_label_dir)
        
        # 효과 분리
        self.night_effects = [
            self.add_wind,
            self.add_thunderstorm,
            self.add_tornado,
            self.add_falling_leaves  # 선택 사항: 밤에 낙엽 허용할지 말지 결정 가능
        ]

        self.day_effects = [
            self.add_wind,
            self.add_rainbow,
            self.add_tornado,
            self.add_falling_leaves
        ]

        self.transform = transform
        self.output_num = output_num
        self.day_or_night = day_or_night.lower()

    def run(self):
        for filename, image, bboxes, class_labels in self.__get_image_data__():
            for index in range(self.output_num):
                # Albumentation 라이브러리로 증강
                augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_image = augmented['image']
                
                # OpenCV 증강: 1~3개 무작위 조합
                # 낮/밤에 따라 사용할 효과 결정
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

    # CV2를 사용한 이미지 증강 예제
    # Albumentation 라이브러리에 있는 기능 제외 CV2를 혼합하여 사용
    # 바람 
    def add_wind(self, img):
        blurred = cv2.GaussianBlur(img, (11, 3), sigmaX=10)
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        windy = cv2.add(blurred, noise)
        return windy
    
    # 번개
    # 야간 + 밝은 섬광 효과 (랜덤한 강한 밝기 영역)
    def add_thunderstorm(self, img):
        img = cv2.convertScaleAbs(img, alpha=0.5, beta=-30)  # 어둡게
        mask = np.zeros_like(img, dtype=np.uint8)
        for _ in range(2):
            x, y = np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0])
            radius = np.random.randint(50, 120)
            cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)
        lightning = cv2.addWeighted(img, 1.0, mask, 0.6, 0)
        return lightning
    
    # 무지개
    # 화려한 색상의 아크를 투명하게 합성
    def add_rainbow(self, img):
        rainbow = np.zeros_like(img)
        h, w = img.shape[:2]
        center = (w // 2, h)
        for i, color in enumerate([(255, 0, 0), (255, 127, 0), (255, 255, 0), 
                                (0, 255, 0), (0, 0, 255), (75, 0, 130), (143, 0, 255)]):
            cv2.ellipse(rainbow, center, (int(w*0.8)-i*8, int(h*0.5)-i*8), 0, 0, 180, color, thickness=6)
        rainbowed = cv2.addWeighted(img, 1.0, rainbow, 0.3, 0)
        return rainbowed

    # 밤
    # 채도와 밝기를 줄이고 파란 빛을 추가
    def add_night(self, img):
        dark = cv2.convertScaleAbs(img, alpha=0.5, beta=-50)
        night = cv2.addWeighted(dark, 1.0, np.full_like(img, (10, 10, 30)), 0.3, 0)
        return night
    
    # 회오리 바람
    def add_tornado(self, img):
        h, w = img.shape[:2]
        
        # 회전 왜곡을 위한 좌표 그리드 생성
        center = (w // 2, h // 2)
        strength = 0.0008  # 회오리 강도 (조절 가능)

        # 좌표 맵 생성
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

        # 회전 왜곡 적용
        swirl = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # 블러와 노이즈로 현실감 추가
        swirl_blur = cv2.GaussianBlur(swirl, (5, 5), sigmaX=3)
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        tornado = cv2.add(swirl_blur, noise)
        
        return tornado
    
    # 하늘에서 떨어지는 것
    def add_falling_leaves(self, img, num_leaves=30):
        h, w = img.shape[:2]
        leaf_img = np.zeros_like(img)

        for _ in range(num_leaves):
            # 낙엽 색상 랜덤
            color = random.choice([(165, 42, 42), (255, 140, 0), (255, 215, 0), (139, 69, 19)])  # 갈색, 주황, 노랑, 짙은 갈색
            
            # 크기, 위치, 회전 각도 랜덤
            center = (random.randint(0, w), random.randint(0, h))
            size = random.randint(10, 30)
            angle = random.uniform(0, 360)
            
            # 타원 그리기 (낙엽 형태)
            # 색상 랜덤
            axes = (size, int(size * 0.4))
            temp = np.zeros_like(img)
            cv2.ellipse(temp, center, axes, angle, 0, 360, color, -1)

            # 누적
            alpha = random.uniform(0.3, 0.7)
            leaf_img = cv2.addWeighted(leaf_img, 1.0, temp, alpha, 0)

         # 낙엽 합성
        result = cv2.addWeighted(img, 1.0, leaf_img, 0.4, 0)
        return result
    
# Albumentation 라이브러리를 사용한 이미지 증강 시퀀스 예제
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
                rain_type='drizzle',
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

# 사용 예시
image_dir = './Image-Detection-2/train/images'
label_dir = './Image-Detection-2/train/labels'
output_image_dir = './syn/syn_img'
output_label_dir = './syn/syn_label'


aug = Augmentator(
    image_dir=image_dir,
    label_dir=label_dir,
    output_image_dir=output_image_dir,
    output_label_dir=output_label_dir,
    transform=transform,
    output_num=1, # 한 이미지로 몇장의 증강 데이터를 만들지 결정
)
aug.run()

synth = ObjectSynthesizer(
    image_dir=image_dir,
    label_dir=label_dir,
    output_image_dir=output_image_dir,
    output_label_dir=output_label_dir,
    obj_num=3, 
)
synth.run()