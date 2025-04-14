import numpy as np
import cv2
import os
import model.Augmentator as Aug
import albumentations as A
from albumentations.core.transforms_interface import DualTransform


# print('aug_test!!!')
image_dir = './test_data/single_data/train/images/'
label_dir = './test_data/single_data/train/labels/'
output_image_dir = './test_data/single_data/train/aug_images/'
output_label_dir = './test_data/single_data/train/aug_labels/'

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

def read_labels(label_path):
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

donor_images = [] # [(img1, bboxes1, labels1), (img2, bboxes2, labels2)]
for filename in os.listdir(image_dir):
    if not filename.endswith('.jpg'):
        continue

    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))
    
    image = cv2.imread(image_path)
    if image is None:
        continue

    bboxes, class_labels = read_labels(label_path)
    donor_images.append((image, bboxes, class_labels))

# print(donor_images)


class CustomCopyPaste(DualTransform):
    def __init__(self, donor_images, obj_num, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.obj_num = obj_num
        self.donor_images = donor_images
    
    def apply(self, img, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        return bbox
    
    def get_transform_init_args_names(self):
        return ("donor_images", )
    
    def __get_random_image__(self):
        idx_list = np.random.choice(len(self.donor_images), size=self.obj_num, replace=False)
        
        random_images = map(lambda x : self.donor_images[x], idx_list)
        return list(random_images)

    def __yolo_to_xxyy__(self, img, bbox):
        xc, yc, w, h = bbox
        ih, iw = img.shape[:2]
        x1 = int((xc - w/2) * iw)
        x2 = int((xc + w/2) * iw)
        y1 = int((yc - h/2) * ih)
        y2 = int((yc - h/2) * ih)
        obj = img[y1:y2, x1:x2] 
        return obj

    def __call__(self, force_apply=False, **data):
        target_img = data['image']
        target_bboxes = data.get("bboxes", [])
        target_labels = data.get("target_labels", [])

        height, width = target_img.shape[:2]

        new_img = target_img.copy()
        new_bboxes = list(target_bboxes)
        new_labels = list(target_labels)
        
        random_img, random_bboxes, random_labels = self.__get_random_image__()

        for bbox, label in zip(random_bboxes, random_labels):
            obj = self.__yolo_to_xxyy__(random_img, bbox)
            oh, ow = obj.shape[:2]

            new_x = np.random.randint(0, width)
            new_y = np.random.randint(0, height)

            new_img[new_y:oh, new_x:ow] = obj

            new_xc = new_x + ow/2
            new_yc = new_y + oh/2
            new_w = ow
            new_y = oh

            new_bboxes.append([new_xc, new_yc, new_w, new_y])
            new_labels.append(label)

        data['image'] = new_img
        data["bboxes"] = new_bboxes
        data["target_labels"] = new_labels

        return data
    
"""
. 기본 이미지 받아오기 (data)
. random으로 객체로 추가할 이미지 정하기 (객체를 몇개 더 추가할지 num 파람 추가)
. 기본 이미지 제외하고 나머지 이미지의 객체 위치 (x1, x2, y1, y2) 구하기: yolo to xxyy
. 기본 이미지 위에 나머지 이미지 랜덤으로 넣기: xxyy to yolo
. 새로 만들어진 이미지 하나 정보 return (img, bboxes, classes)
"""

transform = A.Compose([
    CustomCopyPaste(donor_images=donor_images, obj_num=1, always_apply=True, p=1)
], bbox_params={
    "format": 'yolo',
    "label_fields": ["class_labels"]
})


aug = Aug.Augmentator(
    image_dir=image_dir,
    label_dir=label_dir,
    output_image_dir=output_image_dir,
    output_label_dir=output_label_dir,
    transform=transform,
    output_num=1, # 한 이미지로 몇장의 증강 데이터를 만들지 결정
)
aug.run()