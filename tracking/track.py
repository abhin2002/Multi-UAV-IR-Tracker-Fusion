# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path

import torch
import os

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):

    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)

    yolo = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model)
        else 'yolov8n.pt',
    )
    # Extract video filename without extension
    video_filename = os.path.basename(args.source)  # Get file name from path
    video_name = os.path.splitext(video_filename)[0]  # Remove extension

    # Construct label file path
    label_directory = "/mnt/DATA/Arun/Downloads/MultiUAV_Train/MultiUAV_val/TestLabels_FirstFrameOnly/"
    label_file = os.path.join(label_directory, f"{video_name}.txt")

    # Load first-frame labels
    first_frame_labels = load_first_frame_labels(label_file)
    print("First frame: ",label_file, first_frame_labels)

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    
    

    

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))
    
    if not is_ultralytics_model(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        
        yolo_model = m(model=args.yolo_model, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model

        # If current model is YOLOX, change the preprocess and postprocess
        if is_yolox_model(args.yolo_model):
            # add callback to save image paths for further processing
            yolo.add_callback("on_predict_batch_start",
                              lambda p: yolo_model.update_im_paths(p))
            yolo.predictor.preprocess = (
                lambda imgs: yolo_model.preprocess(im=imgs))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    
    # store custom args in predictor
    yolo.predictor.custom_args = args
    
        

    with open(f"/mnt/DATA/Arun/Downloads/MultiUAV_Train/boxmot-strongsort/boxmot/tracking/result-track/{video_name}.txt", 'w') as f:
        for i, r in enumerate(results):
            # Replace first frame detections with predefined labels
            if i == 0:
                print("~~~~~~~~~~~~~~~~~~~")
                r.boxes = []
                for det in first_frame_labels:
                    box = Box()
                    box.id = torch.tensor(det["id"])
                    box.xyxy = torch.tensor([det["bbox"]])
                    box.class_id = torch.tensor(det["class_id"])
                    box.conf = torch.tensor(det["confidence"])
                    r.boxes.append(box)
            
            # Generate image with tracking results
            img = r.orig_img  # Simply keep the original image without modification

            
            # Display image if enabled
            if args.show:
                cv2.imshow('BoxMOT', img)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord(' '), ord('q')):
                    break
            
            # Save tracking results in MOTChallenge format
            for box in r.boxes:
                if box.id is None:
                    print(f"Warning: Skipping box with missing ID at frame {i+1}")
                    continue  # Skip this box if ID is None

                bbox = box.xyxy[0].tolist()  # Convert from tensor to list
                track_id = box.id.item()  # Get track id safely
                conf = box.conf.item()  # Get confidence score
                f.write(f'{i+1},{track_id},{bbox[0]},{bbox[1]},{bbox[2]-bbox[0]},{bbox[3]-bbox[1]},-1,-1,{conf}\n')


def load_first_frame_labels(label_path):
    """
    Reads the first frame labels from a file.
    Expected format: frame_id, obj_id, x, y, w, h, conf, class_id, visibility
    """
    first_frame_detections = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue  # Ensure correct format

            frame_id, obj_id, x, y, w, h, conf, class_id, visibility = map(float, parts)
            
            # Convert (x, y, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = x, y, x + w, y + h

            first_frame_detections.append({
                "id": int(obj_id),
                "bbox": [x1, y1, x2, y2],  # Now in (x1, y1, x2, y2) format
                "class_id": int(class_id),
                "confidence": conf
            })

    return first_frame_detections


def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None,
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')

    opt = parser.parse_args()
    return opt

class Box:
    def __init__(self):
        self.id = None
        self.xyxy = None
        self.class_id = None
        self.conf = None

if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
