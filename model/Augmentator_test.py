import os
import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform

# 이미지 정보 리턴 클래스: filename, images, bboxes, class_labels
class ImageMeta:
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir

    def __read_labels__(self, label_path):
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

    def __get_image_data__(self):
        image_meta_list = [] # [(img1, bboxes1, labels1), (img2, bboxes2, labels2)]
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
        return image_meta_list

# 이미지 객체 합성 클래스
class ObjectSynthesizer(DualTransform, ImageMeta):
    def __init__(self, image_dir, label_dir, obj_num, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.image_dir = image_dir # 추가할 객체가 있는 이미지 dir
        self.label_dir = label_dir # 추가할 객체가 있는 라벨 dir
        self.obj_num = obj_num # 기본 이미지에 추가할 객체 개수
    

    def targets_as_params(self):
        print("targets_as_params")
        return [
            "obj_img",
            "obj_bboxes",
            "obj_labels",
        ]

    def get_params_dependent_on_target(self, params):
        print("get_params_dependent_on_target >>>", params)
        obj_img, obj_bboxes, obj_labels = self.__get_random_image__()

        return {
            "obj_img": obj_img,
            "obj_bboxes": obj_bboxes,
            "obj_labels": obj_labels
        }
    
    def apply(self, img, **params):
        print("apply", img, params)
        return img
    
    def apply_to_bbox(self, bbox, **params):
        print("apply_to_bbox", bbox, params)
        return bbox

    # def apply_to_bbox(self, bbox, target_img, obj_img, **params):
    #     print("apply_to_bbox", bbox, params)
    #     new_img = target_img.copy()

    #     target_h, target_w = target_img.shape[:2]

    #     obj, obj_w, obj_h = self.__get_obj_info__(obj_img, bbox)

    #     max_x = target_w - obj_w
    #     max_y = target_h - obj_h

    #     new_x = np.random.randint(0, max_x)
    #     new_y = np.random.randint(0, max_y)

    #     new_img[new_y:new_y+obj_h, new_x:new_x+obj_w] = obj
    #     self.apply(new_img)

    #     # yolo 형식에 맞게 변환
    #     new_bbox = self.__get_yolo_data__(new_x, obj_w, target_w, new_y, obj_h, target_h)

    #     return new_bbox
    

    def get_transform_init_args_names(self):
        print("get_transform_init_args_names")
        return ("image_dir", "label_dir", "obj_num")
    
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
        new_xc = (new_x + obj_w/2) / target_w
        new_yc = (new_y + obj_h/2) / target_h
        new_w = obj_w / target_w
        new_h = obj_h / target_h
        return (new_xc, new_yc, new_w, new_h)


    def __call__(self, force_apply=False, **data):
        target_img = data["image"]
        target_bboxes = data.get("bboxes", [])
        target_labels = data.get("class_labels", [])
        # self.get_params_dependent_on_data()

        # new_img = target_img.copy()
        new_bboxes = []
        new_labels = list(target_labels)

        for x_min, y_min, x_max, y_max, class_label in target_bboxes:
            new_bbox = self.__get_yolo_data__(x_min, x_max-x_min, 1, y_min, y_max-y_min, 1)
            new_bboxes.append(new_bbox)
        
        obj_img, obj_bboxes, obj_labels = self.__get_random_image__()

        for bbox, label in zip(obj_bboxes, obj_labels):
            new_bbox = self.apply_to_bbox(bbox, target_img, obj_img)
            # 랜덤으로 얻은 객체를 랜덤위치에 붙여넣기
            new_bboxes.append(new_bbox)
            new_labels.append(label)
            # print("??", new_bboxes, new_labels)

            # data["image"] = new_img
            data["bboxes"] = new_bboxes
            data["class_labels"] = new_labels

        return data

# 이미지 증강 후 저장 클래스 
class Augmentator(ImageMeta):
    def __init__(self, image_dir, label_dir, output_image_dir, output_label_dir, transform, output_num=1):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_image_dir = output_image_dir
        self.output_label_dir = output_label_dir
        self.transform = transform
        self.output_num = output_num

    def run(self):
        for filename, image, bboxes, class_labels in self.__get_image_data__():
            for index in range(self.output_num):
                augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                self.__save_output__(filename, augmented, index)

    def __save_output__(self, filename, augmented, index):
        print("++++", augmented)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        # 증강 이미지 저장
        output_images_path = os.path.join(self.output_image_dir, f'{filename}_{index}.jpg')
        output_labels_path = os.path.join(self.output_label_dir, f'{filename}_{index}.txt')
        # print("@@@", output_labels_path)

        cv2.imwrite(output_images_path, aug_image)
        with open(output_labels_path, 'w') as f:
            for label, bbox in zip(aug_labels, aug_bboxes):
                x, y, w, h = bbox  # 이미 YOLO 형식일 경우
                # print("!!!", label, x, y, w, h)
                f.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    
# 객체 합성 transform 예제
image_dir = './test_data/single_data/train/images/'
label_dir = './test_data/single_data/train/labels/'

output_image_dir = './test_data/single_data/train/aug_images/'
output_label_dir = './test_data/single_data/train/aug_labels/'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

transform_synth = A.Compose(
    [
        # ObjectSynthesizer( 
        #     always_apply=True, 
        #     p=1
        # )
        ObjectSynthesizer( 
            image_dir=image_dir,
            label_dir=label_dir,
            obj_num=3, 
            always_apply=True, 
            p=1
        )
    ], 
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

aug = Augmentator(
    image_dir=image_dir,
    label_dir=label_dir,
    output_image_dir=output_image_dir,
    output_label_dir=output_label_dir,
    transform=transform_synth,
    # transform=Aug.transform,
    output_num=1, # 한 이미지로 몇장의 증강 데이터를 만들지 결정
)
aug.run()



# 이미지 증강 기법 시퀀스 예제
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