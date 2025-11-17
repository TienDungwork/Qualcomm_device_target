import sys
import os
import cv2
import numpy as np
import time

# Add virtual_fence to path for DetectionQNN
sys.path.insert(0, os.path.dirname(__file__))

from virtual_fence.yolov5.detect_qnn import DetectionQNN
from virtual_fence.yolov5.setup import read_model_config_file

def infer_and_draw(input_img_path, output_img_path="output.jpg"):
    # Instead of using setup.json, we directly specify yolov8_det1.bin and dummy defaults for QNN
    # Read model config for another config? No, as per instructions, directly load yolov8_det1.bin
    
    # COCO class names (80 classes) - update this if your model uses different classes
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # You may need to update this path to the correct model location
    yolov8_bin_path = os.path.join(os.path.dirname(__file__), "yolov8_det1.bin")  # Update if needed
    
    # Filter classes: None = all classes, [0] = only person, [0,2,5,7] = person, car, bus, truck
    qnn_classes = None  # Change this to [0] to detect only person, or None for all classes
    conf = 0.5
    imgsz = 640
    device = "cpu"     # Not used for QNN but argument is present
    data = ""          # Not used for QNN
    qnn_sdk_path = "/home/ntiendung/qairt/2.40.0.251030/lib/aarch64-oe-linux-gcc11.2"  # Could also be provided externally

    # Initialize DetectionQNN
    try:
        detector = DetectionQNN()
        detector.qnn_sdk_path = qnn_sdk_path
        detector.setup_model(yolov8_bin_path, qnn_classes, conf, imgsz, device, data)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return

    # Load the test image
    img = cv2.imread(input_img_path)
    if img is None:
        print(f"Failed to load image {input_img_path}")
        return

    # Warm up 10 times (no timing)
    print("Warming up QNN detector for 10 runs...")
    for _ in range(10):
        _ = detector.detect(img)

    # Only calculate time for bboxes = detector.detect(img)
    infer_times = []
    print("Collecting 10 inference runs for timing...")

    for i in range(10):
        t0 = time.time()
        bboxes = detector.detect(img)
        t1 = time.time()
        print(f"Inference time {i}: {t1 - t0:.6f} seconds")
        infer_times.append(t1 - t0)

    avg_inf = sum(infer_times) / len(infer_times)
    print(f"Average Inference (detector.detect): {avg_inf*1000:.2f} ms")

    # Do one last (measured) full detect, for drawing and saving the box
    try:
        bboxes = detector.detect(img)
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # Draw bounding boxes
    for bbox in bboxes:
        # From detect_qnn.py, bbox format is [x1, y1, x2, y2, class_id, confidence]
        # We'll match those
        x1, y1, x2, y2 = map(int, bbox[:4])
        cls = int(bbox[4]) if len(bbox) > 4 else -1
        conf = bbox[5] if len(bbox) > 5 else 0

        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Show class name instead of just ID
        class_name = COCO_CLASSES[cls] if 0 <= cls < len(COCO_CLASSES) else f"ID:{cls}"
        label = f"{class_name} {conf:.2f}"
        cv2.putText(img, label, (x1, max(y1-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the output image
    cv2.imwrite(output_img_path, img)
    print(f"Inference complete. Output saved to {output_img_path}")

if __name__ == "__main__":
    # These environment variables are just examples. Modify these paths as necessary.
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QNN_SDK_ROOT"] = "/home/ntiendung/qairt/2.40.0.251030"
    os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + \
        "/home/ntiendung/qairt/2.40.0.251030/lib/aarch64-oe-linux-gcc11.2"
    os.environ["ADSP_LIBRARY_PATH"] = "/home/ntiendung/qairt/2.40.0.251030/lib/hexagon-v73/unsigned"

    # Run inference on test.jpg using yolov8_det1.bin
    infer_and_draw("test.jpg", "output.jpg")
