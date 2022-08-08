# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


class Yolo_detect_wuzi():
    def __init__(self,weights='weight/last.pt',device='',half=False,imgsz=640):
        # Initialize
        self.half = half

        set_logging()
        self.imgsz=imgsz
        self.device = select_device(device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA
        print("yolo  init")
        self.model = attempt_load(weights, map_location=self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.stride = int(self.model.stride.max())  # model stride
        if self.half:
            self.model.half()  # to FP16
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz,self.imgsz).to(self.device).type_as(next(self.model.parameters())))

        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

    @torch.no_grad()
    def run(self,img0,
            conf_thres=0.1,  # confidence threshold
            iou_thres=0.5,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=True,  # augmented inference
            visualize=False,  # visualize features
            ):

        phoneBboxes = []
        # 矩形图片信封填充为正方形
        img = letterbox(img0,self.imgsz, self.stride,True)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        dataset=[[None, img, img0, None]]

        # Run inference
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # Inference
            pred = self.model(img, augment=augment, visualize=visualize)[0]
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            pred_boxes = []
            for det in pred:
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    for *x, conf, cls_id in det:

                        lbl = self.names[int(cls_id)]
                        x1, y1 = int(x[0]), int(x[1])
                        x2, y2 = int(x[2]), int(x[3])
                        conf = conf.cpu().numpy()
                        conf = round(conf.tolist(), 4)
                        pred_boxes.append((x1, y1, x2, y2, lbl, conf))
            phoneBboxes.append(pred_boxes)
        return phoneBboxes


