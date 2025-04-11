import os
import cv2
import albumentations as A

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
            # ☁️ 안개
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.6, alpha_coef=0.1, p=1.0),

            # 🌧️ 비
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

            # ❄️ 눈
            A.RandomSnow(
                snow_point_lower=0.1,
                snow_point_upper=0.3,
                brightness_coeff=1.2,
                p=1.0
            ),

            # 🌞 햇빛/플레어
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),  # 상단 50% 영역에만 적용
                angle_lower=0.3,
                src_radius=100,
                src_color=(255, 255, 255),
                p=1.0
            ),

            # 🌑 그림자 (나무, 건물 등)
            A.RandomShadow(
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                shadow_roi=(0, 0.5, 1, 1),  # 하단 절반에만 그림자
                p=1.0
            ),
        ], p=0.3),
        # ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)


class Augmentator:
    def __init__(self, image_dir, label_dir, output_image_dir, output_label_dir, transform, output_num=1):
        self.output_num = output_num
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_image_dir = output_image_dir
        self.output_label_dir = output_label_dir
        self.transform = transform

    def run(self):
        for filename in os.listdir(self.image_dir):
            if not filename.endswith('.jpg'):
                continue

            # transform - image 
            image_path = os.path.join(self.image_dir, filename)
            label_path = os.path.join(self.label_dir, filename.replace('.jpg', '.txt'))
            
            image = cv2.imread(image_path)
            if image is None:
                continue

            bboxes, class_labels = self.__read_label__(label_path)

            # 이미지 증강
            for index in range(self.output_num):
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                self.__save_output__(filename, augmented, index)
            
    def __read_label__(self, label_path):
        bboxes = []
        class_labels = []

        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = parts[0]  # 클래스 ID
                bbox = [float(i) for i in parts[1:]] # [x, y, width, height]
                
                bboxes.append(bbox)
                class_labels.append(int(float(class_id)))
        return bboxes, class_labels
    
    def __save_output__(self, filename, augmented, index):
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        # 증강 이미지 저장
        output_images_path = os.path.join(self.output_image_dir, f'{filename}_{index}.jpg')
        output_labels_path = os.path.join(self.output_label_dir, f'{filename}_{index}.txt')

        cv2.imwrite(output_images_path, aug_image)
        with open(output_labels_path, 'w') as f:
            for label, bbox in zip(aug_labels, aug_bboxes):
                x, y, w, h = bbox  # 이미 YOLO 형식일 경우
                f.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    