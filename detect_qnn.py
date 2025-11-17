"""
YOLOv8 Detection using Qualcomm QNN (Quantized Neural Network)
Adapted from ai-engine-direct-helper yolov8_det.py
"""

import time
import numpy as np
from qai_appbuilder import (QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig)
import cv2
import torchvision.transforms as transforms
import torch
from torchvision.ops import nms



def relu(x):
    return max(0, int(x))


class YoloV8QNN(QNNContext):
    """YOLOv8 class inherited from QNNContext"""
    def Inference(self, input_data):
        input_datas = [input_data]
        output_data = super().Inference(input_datas)
        return output_data


class DetectionQNN:
    """
    YOLOv8 Detection using QNN backend
    Compatible interface with existing Detection class
    """
    def __init__(self):
        self.model = None
        self.model_file = ''
        self.imgsz = 640
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.classes = [0]
        self.qnn_sdk_path = ""
        
    def setup_model(self, a_model, classes, conf_thres, img_size, device, data):
        """
        Setup QNN model
        
        Args:
            a_model: Path to .bin model file
            classes: List of class IDs to detect
            conf_thres: Confidence threshold
            img_size: Input image size
            device: Device (not used for QNN, but kept for compatibility)
            data: Data yaml path (not used for QNN)
        """
        self.conf_thres = conf_thres
        self.imgsz = img_size
        self.model_file = a_model
        self.classes = classes
        
        # Config AppBuilder environment
        # Default QNN SDK path - should be configured in setup.json
        if not self.qnn_sdk_path:
            self.qnn_sdk_path = "/home/ntiendung/qairt/2.40.0.251030/lib/aarch64-oe-linux-gcc11.2"
        
        QNNConfig.Config(self.qnn_sdk_path, Runtime.HTP, LogLevel.ERROR, ProfilingLevel.BASIC)
        
        # Initialize YoloV8 model
        self.model = YoloV8QNN("yolov8", str(self.model_file))
        print(f"QNN Model loaded: {self.model_file}")
    
    def detect(self, image):
        """
        Run detection on image
        
        Args:
            image: numpy array (H, W, C) BGR format
            
        Returns:
            bboxes: List of [x1, y1, x2, y2, class_id, confidence]
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        
        # Preprocess image
        t = time.time()
        img = self._preprocess(image)
        t1 = time.time()
        print(f"Preprocess time: {t1 - t:.6f} seconds")
        t = time.time()
        # Burst the HTP profile, log level off, basic profiling
        # PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
        t2 = time.time()
        print(f"Set Perf Profile time: {t2 - t1:.6f} seconds")
        t = time.time()
        # Run inference
        model_output = self.model.Inference([img])
        t3 = time.time()
        print(f"Inference time: {t3 - t2:.6f} seconds")
        # # Reset the HTP
        # PerfProfile.RelPerfProfileGlobal()
        t4 = time.time()
        print(f"Postprocess time: {t4 - t3:.6f} seconds")
        # Check if inference was successful
        
        if not model_output or len(model_output) < 3:
            print("Warning: Model inference failed or returned invalid output")
            return []
        
        # Parse output
        bboxes = self._postprocess(model_output, image.shape)
        t5 = time.time()
        print(f"Postprocess time: {t5 - t4:.6f} seconds")
        return bboxes
    
    def _preprocess(self, image):
        """
        Preprocess image for YOLOv8 QNN
        
        Args:
            image: numpy array (H, W, C) BGR format
            
        Returns:
            Preprocessed image ready for inference
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size using OpenCV
        resized_img = cv2.resize(image_rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img_np = resized_img.astype(np.float32) / 255.0  # 0-255 to 0.0-1.0

        # Add batch dimension and permute to NHWC format (1, H, W, 3)
        img = np.expand_dims(img_np, axis=0)
        
        return img

    def _postprocess(self, model_output, original_shape):
        """
        Postprocess model output to get bounding boxes
        
        Args:
            model_output: Raw output from QNN model
            original_shape: Original image shape (H, W, C)
            
        Returns:
            bboxes: List of [x1, y1, x2, y2, class_id, confidence]
        """
        
        # Parse model output
        pred_boxes = torch.tensor(model_output[0].reshape(1, -1, 4))
        pred_scores = torch.tensor(model_output[1].reshape(1, -1))
        pred_class_idx = torch.tensor(model_output[2].reshape(1, -1))
        
        # Apply NMS
        boxes_out, scores_out, class_idx_out = self._batched_nms(
            self.iou_thres,
            self.conf_thres,
            pred_boxes,
            pred_scores,
            pred_class_idx,
        )
        
        # Convert to bboxes format
        bboxes = []
        H, W = original_shape[:2]
        scale_x = W / self.imgsz
        scale_y = H / self.imgsz
        
        for batch_idx in range(len(boxes_out)):
            pred_boxes_batch = boxes_out[batch_idx]
            pred_scores_batch = scores_out[batch_idx]
            pred_class_idx_batch = class_idx_out[batch_idx]
            
            for box, score, class_idx in zip(pred_boxes_batch, pred_scores_batch, pred_class_idx_batch):
                cls = int(class_idx.item())
                
                # Filter by class if specified
                if self.classes and cls not in self.classes:
                    continue
                
                # Scale boxes back to original size
                x1 = int(box[0].item() * scale_x)
                y1 = int(box[1].item() * scale_y)
                x2 = int(box[2].item() * scale_x)
                y2 = int(box[3].item() * scale_y)
                
                # Clamp to image boundaries
                x1 = max(0, min(x1, W))
                y1 = max(0, min(y1, H))
                x2 = max(0, min(x2, W))
                y2 = max(0, min(y2, H))
                
                conf = float(score.item())
                
                bboxes.append([x1, y1, x2, y2, cls, conf])
        
        return bboxes
    
    @staticmethod
    def _batched_nms(iou_threshold, score_threshold, boxes, scores, *gather_additional_args):
        """
        Non maximum suppression over several batches
        """
        from torchvision.ops import nms
        import torch
        from typing import List
        
        scores_out: List[torch.Tensor] = []
        boxes_out: List[torch.Tensor] = []
        args_out: List[List[torch.Tensor]] = (
            [[] for _ in gather_additional_args] if gather_additional_args else []
        )
        
        for batch_idx in range(0, boxes.shape[0]):
            # Clip outputs to valid scores
            batch_scores = scores[batch_idx]
            scores_idx = torch.nonzero(scores[batch_idx] >= score_threshold).squeeze(-1)
            batch_scores = batch_scores[scores_idx]
            batch_boxes = boxes[batch_idx, scores_idx]
            batch_args = (
                [arg[batch_idx, scores_idx] for arg in gather_additional_args]
                if gather_additional_args
                else []
            )
            
            if len(batch_scores) > 0:
                nms_indices = nms(batch_boxes[..., :4], batch_scores, iou_threshold)
                batch_boxes = batch_boxes[nms_indices]
                batch_scores = batch_scores[nms_indices]
                batch_args = [arg[nms_indices] for arg in batch_args]
            
            boxes_out.append(batch_boxes)
            scores_out.append(batch_scores)
            for arg_idx, arg in enumerate(batch_args):
                args_out[arg_idx].append(arg)
        
        return boxes_out, scores_out, *args_out


# Alias for compatibility
Detection = DetectionQNN
