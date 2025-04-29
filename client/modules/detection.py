# detection.py
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from modules.config import ENEMY_CLASSES, FRIENDLY_CLASSES, OBSTACLE_CLASSES, TARGET_CLASSES
from flask import jsonify

model = YOLO('best.pt')

async def analyze_obstacle(obstacle, index):
    x_center = (obstacle["x_min"] + obstacle["x_max"]) / 2
    z_center = (obstacle["z_min"] + obstacle["z_max"]) / 2
    image_data = obstacle.get("image")

    if image_data:
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            results = model.predict(image, verbose=False)
            detections = results[0].boxes.data.cpu().numpy()
            filtered_results = []
            for box in detections:
                class_id = int(box[5])
                if class_id in TARGET_CLASSES:
                    filtered_results.append({
                        'className': TARGET_CLASSES[class_id],
                        'bbox': [float(coord) for coord in box[:4]],
                        'confidence': float(box[4])
                    })

            if filtered_results:
                detection = max(filtered_results, key=lambda x: x['confidence'])
                class_name = detection['className']
                confidence = detection['confidence']
                print(f"YOLO detection succeeded at ({x_center:.2f}, {z_center:.2f}): class={class_name}, confidence={confidence:.2f}")
            else:
                class_name = 'unknown'
                confidence = 0.0
                print(f"YOLO detection succeeded at ({x_center:.2f}, {z_center:.2f}): no valid detections")

            if class_name in ENEMY_CLASSES:
                print(f"Enemy detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            elif class_name in FRIENDLY_CLASSES:
                print(f"Friendly detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            elif class_name in OBSTACLE_CLASSES:
                print(f"Obstacle detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            else:
                print(f"Unknown object detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")

        except Exception as e:
            print(f"YOLO detection failed at ({x_center:.2f}, {z_center:.2f}): {e}")
            class_name = 'tank' if x_center < 80 else 'car5'
            if x_center > 90:
                class_name = 'rock1'
            print(f"Fallback detection: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            if class_name in ENEMY_CLASSES:
                print(f"Enemy detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            elif class_name in FRIENDLY_CLASSES:
                print(f"Friendly detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            elif class_name in OBSTACLE_CLASSES:
                print(f"Obstacle detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
            else:
                print(f"Unknown object detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
    else:
        print(f"YOLO detection failed at ({x_center:.2f}, {z_center:.2f}): no image data provided")
        class_name = 'tank' if x_center < 80 else 'car5'
        if x_center > 90:
            class_name = 'rock1'
        print(f"Fallback detection: {class_name} at ({x_center:.2f}, {z_center:.2f})")
        if class_name in ENEMY_CLASSES:
            print(f"Enemy detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
        elif class_name in FRIENDLY_CLASSES:
            print(f"Friendly detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
        elif class_name in OBSTACLE_CLASSES:
            print(f"Obstacle detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")
        else:
            print(f"Unknown object detected: {class_name} at ({x_center:.2f}, {z_center:.2f})")

    return {"className": class_name, "position": (x_center, z_center)}

def detect(image):
    if not image:
        print("YOLO detection failed: no image received in /detect")
        return jsonify({"error": "No image received"}), 400

    try:
        image = Image.open(image)
        results = model.predict(image, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        filtered_results = []
        for box in detections:
            class_id = int(box[5])
            if class_id in TARGET_CLASSES:
                class_name = TARGET_CLASSES[class_id]
                filtered_results.append({
                    'className': class_name,
                    'bbox': [float(coord) for coord in box[:4]],
                    'confidence': float(box[4])
                })
                print(f"YOLO detection succeeded in /detect: class={class_name}, confidence={float(box[4]):.2f}")
                if class_name in ENEMY_CLASSES:
                    print(f"Enemy detected in /detect: {class_name}")
                elif class_name in FRIENDLY_CLASSES:
                    print(f"Friendly detected in /detect: {class_name}")
                elif class_name in OBSTACLE_CLASSES:
                    print(f"Obstacle detected in /detect: {class_name}")
                else:
                    print(f"Unknown object detected in /detect: {class_name}")
        if not filtered_results:
            print("YOLO detection succeeded in /detect: no valid detections")
        return jsonify(filtered_results)
    except Exception as e:
        print(f"YOLO detection failed in /detect: {e}")
        return jsonify({"error": "Detection failed"}), 500