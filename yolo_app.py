import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

track_history = defaultdict(list)

current_region = None
counting_region = [
    {
        "name": "Counting Region",
        "polygon": Polygon([(440, 91), (510, 89), (465, 358), (217, 357)]),  # Polygon points
        "counts": 0,
        "draggin": False,
        "region_color": (37, 255, 255), # BGR value
        "text_color": (0, 0, 0) # Region text color
    }
]

def run(
        weights="yolov10m.pt",
        source=None,
        device="gpu",
        view_img=False,
        save_img=False,
        exist_ok=False,
        classes=None,
        line_thickness=2,
        track_thickness=2,
        region_thickness=2
):
    """
    Run Region counting on a video using YOLOv10 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.
    Regions can be Polygons or rectangle in shape

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        device (str): processing device cpu, 0, 1
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        classes (list): classes to detect and track
        line_thickness (int): Bounding box thickness.
        track_thickness (int): Tracking line thickness
        region_thickness (int): Region thickness.
    """
    # vid_frame_count = 0
    
    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")
    
    # Setup Model
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    # Extract class names
    names = model.model.names

    # Video Setup
    VideoCapture = cv2.VideoCapture(source)
    frame_w, frame_h, fps = (int(VideoCapture.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, 
                                                                cv2.CAP_PROP_FRAME_HEIGHT, 
                                                                cv2.CAP_PROP_FPS
                                       ))
    save_dir = increment_path(Path("counter_region_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    codec = "mp4v"
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"),
                        cv2.VideoWriter_fourcc(*codec), fps, (frame_w,frame_h))
    
    # Iterate over video frames
    prev_time = 0
    while VideoCapture.isOpened():
        sucess, frame = VideoCapture.read()
        if not sucess:
            continue
        # vid_frame_count += 1

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"{fps:.2f} fps", (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        # Extract the results
        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                # annotator.box_label(box, str(names[cls]), color=colors(cls,True)) #label deteksi
                # annotator.box_label(box, color=colors(cls,True)) # kotak deteksi
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                # track = track_history[track_id] # Tracking lines plot
                # track.append((float(bbox_center[0]), float(bbox_center[1])))
                # if len(track) > 30:
                #     track.pop(0)
                # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # cv2.polylines(frame, [points], isClosed=False,
                #               color=colors(cls,True), thickness=track_thickness)
                
                # Check if detection inside region
                for region in counting_region:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        annotator.box_label(box, color=colors(cls,True)) # kotak deteksi yg didalam region
                        region["counts"] += 1
                
        # Draw regions (Polygons/Rectangles)
        for region in counting_region:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ =cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x -5, text_y - text_size[1]-5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)
            
        if view_img:
            cv2.imshow("Crowd Counter POC", frame)
        
        if save_img:
            video_writer.write(frame)
        
        for region in counting_region: # Reinitialize counter 
            region["counts"] = 0
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # del vid_frame_count
    video_writer.release()
    VideoCapture.release()
    cv2.destroyAllWindows()

def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--track-thickness", type=int, default=2, help="Tracking line thickness")
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")

    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
