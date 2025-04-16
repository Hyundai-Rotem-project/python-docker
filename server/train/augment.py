import os
import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform

# 이미지, 라벨벨 읽기 및 저장
class ImageFileManager:
    def __init__(self, image_dir, label_dir, output_image_dir, output_label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_image_dir = output_image_dir
        self.output_label_dir = output_label_dir

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

        output_images_path = os.path.join(self.output_image_dir, f'{filename}_{index}.jpg')
        output_labels_path = os.path.join(self.output_label_dir, f'{filename}_{index}.txt')

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

        obj = img[y1:y2, x1:x2] 
        obj_w = x2 - x1
        obj_h = y2 - y1

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
    def __init__(self, image_dir, label_dir, output_image_dir, output_label_dir, transform, output_num=1):
        super().__init__(image_dir, label_dir, output_image_dir, output_label_dir)
        self.transform = transform
        self.output_num = output_num

    def run(self):
        for filename, image, bboxes, class_labels in self.__get_image_data__():
            for index in range(self.output_num):
                augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                data = (augmented['image'], augmented['bboxes'], augmented['class_labels'])
                self.__save_output__(filename, data, index)

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

# 사용 예시
image_dir = './test_data/single_data/train/images/'
label_dir = './test_data/single_data/train/labels/'
output_image_dir = './test_data/single_data/train/aug_images/'
output_label_dir = './test_data/single_data/train/aug_labels/'

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

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