import random
import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform

class CustomCopyPaste(DualTransform):
    def __init__(self, donor_images: list, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.donor_images = donor_images  # [(image, bboxes, class_labels)]

    def apply(self, image, **params):
        # 원본 이미지 반환 (실제 변형은 __call__에서 처리)
        return image

    def apply_to_bbox(self, bbox, **params):
        # bbox는 __call__에서 직접 처리하므로 여기에선 pass
        return bbox

    def get_transform_init_args_names(self):
        return ("donor_images",)

    def __call__(self, force_apply=False, **data):
        target_img = data["image"]
        target_bboxes = data.get("bboxes", [])
        target_labels = data.get("class_labels", [])

        height, width = target_img.shape[:2]

        new_img = target_img.copy()
        new_bboxes = list(target_bboxes)
        new_labels = list(target_labels)

        # 랜덤하게 donor 이미지 하나 고르기 (자기 자신 제외)
        donors = [item for item in self.donor_images if not np.array_equal(item[0], target_img)]
        if not donors:
            return data  # 합성할 donor가 없음

        donor_img, donor_bboxes, donor_labels = random.choice(donors)

        for bbox, label in zip(donor_bboxes, donor_labels):
            # YOLO 포맷: [x_center, y_center, w, h] → 절대 좌표 변환
            x_c, y_c, w, h = bbox
            x1 = int((x_c - w / 2) * donor_img.shape[1])
            y1 = int((y_c - h / 2) * donor_img.shape[0])
            x2 = int((x_c + w / 2) * donor_img.shape[1])
            y2 = int((y_c + h / 2) * donor_img.shape[0])
            obj_crop = donor_img[y1:y2, x1:x2]

            # 붙여넣을 위치 (랜덤)
            paste_w, paste_h = x2 - x1, y2 - y1
            max_x = width - paste_w
            max_y = height - paste_h
            if max_x <= 0 or max_y <= 0:
                continue  # 공간이 부족해서 패스

            new_x = random.randint(0, max_x)
            new_y = random.randint(0, max_y)

            # 붙여넣기 (단순한 Copy-Paste)
            new_img[new_y:new_y+paste_h, new_x:new_x+paste_w] = obj_crop

            # 새로운 bbox 좌표 → YOLO 형식으로 변환
            new_x_center = (new_x + paste_w / 2) / width
            new_y_center = (new_y + paste_h / 2) / height
            new_w = paste_w / width
            new_h = paste_h / height

            new_bboxes.append([new_x_center, new_y_center, new_w, new_h])
            new_labels.append(label)

        data["image"] = new_img
        data["bboxes"] = new_bboxes
        data["class_labels"] = new_labels

        return data
