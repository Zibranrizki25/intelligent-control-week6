from ultralytics import YOLO
import cv2

# Load model YOLOv8 Instance Segmentation
model = YOLO("yolov8n-seg.pt")

def detect_rail_lane(image_path):
    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""
    results = model(image_path, show=True)
    results[0].save("lane_detection_result.jpg")

# Contoh penggunaan
detect_rail_lane("rail_segmentation/test/images/1000195092_0014-0_jpeg.rf.bdcc36fa5199304696bb201c6db9c07c.jpg")
