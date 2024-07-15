
# Video Analytics : People Counter Surveillance Using YOLO



![Alt Text](Crowd.gif)

'''
## Deployment

To deploy this project run

```bash
  python "yolo_app_accumulate.py" --source "Crowd.mp4" --view-img --save-img --device 0 --weights "yolov8s.pt" --classes 0 --counter-accumulated
```

parameters
```
"--weights", type=str, default="yolov8s.pt", help="initial weights path"
"--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
"--source", type=str, required=True, help="video file path"
"--view-img", action="store_true", help="show results"
"--save-img", action="store_true", help="save results"
"--exist-ok", action="store_true", help="existing project/name ok, do not increment"
"--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3"
"--line-thickness", type=int, default=2, help="bounding box thickness"
"--region-thickness", type=int, default=4, help="Region thickness")
"--counter-accumulated", action="store_true", help="accumulated counter"
```