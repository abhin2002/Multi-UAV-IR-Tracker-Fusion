
# 🛸 Multi-UAV-IR-Tracker-Fusion

**Multi-UAV-IR-Tracker-Fusion** is a customized extension of [BOXMot](https://github.com/mikel-brostrom/boxmot) designed for **multi-object tracking in UAV-based infrared (IR) videos**. It includes:

- First-frame label-based identity assignment
- deepocsort
  strongsort
  ocsort
  bytetrack
  botsort
  imprassoc
  boosttrack
             -based tracking
- YOLOv8 object detection
- Multi-stream/fusion-ready support

Ideal for **aerial surveillance**, **thermal drone analytics**, and **research in IR video-based tracking**.

---
### Sample Tracking Demo

<div align="center">
  <p>
  <img src="[https://github.com/mikel-brostrom/boxmot/releases/download/v12.0.0/output_640.gif](https://github.com/abhin2002/Multi-UAV-IR-Tracker-Fusion/blob/cb5a386e2ae07e494562eae33e4823acedfd7a63/assets/output_visualized_154.mp4)" width="400"/>
  </p>
  <br>
<div>


## 🚀 Key Features

- ✅ **YOLOv8-based Detection:** High-accuracy object detection
- ✅ **StrongSORT Tracking:** Appearance-based deep re-identification tracking
- ✅ **First-frame Label Initialization:** Uses ground truth from first frame to assign consistent IDs
- ✅ **Infrared Video Support:** Designed for thermal IR UAV datasets
- ✅ **Multi-Video Batch Tracking:** Automatically processes multiple `.mp4`, `.avi`, `.mkv` files


---

## 📁 Project Structure

```
Multi-UAV-IR-Tracker-Fusion/
├── tracking/                    # Tracking module (StrongSORT, Deep ReID, Kalman)
├── yolov8/                      # YOLOv8 detector
├── test_videos/                # Input UAV IR videos (mp4/avi/mkv)
├── first_frame_labels/         # First frame GT labels for ID initialization
├── weights/                    # Pretrained weights
├── outputs/                    # Tracked results, videos, and logs
├── track.py                    # Main tracking entry point
├── requirements.txt            # Python dependencies
└── README.md                   # You're here :)
```

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Multi-UAV-IR-Tracker-Fusion.git
   cd Multi-UAV-IR-Tracker-Fusion
   ```

2. Set up the environment:
   ```bash
   conda create -n boxmot python=3.10
   conda activate boxmot
   pip install -r requirements.txt
   ```

3. Place pretrained weights:
   - YOLOv8: `/path/to/yolov8.pt`
   - ReID: `/weights/osnet_x0_25_msmt17.pt`

---

## 📹 Usage Example

### 🧪 Batch Run on Multiple Videos

```bash
for video in /path/to/MultiUAV_val/*.{mp4,avi,mkv}; do
  python track.py     --yolo-model /path/to/yolov8.pt     --source "$video"     --save     --tracking-method strongsort     --device 0
done
```

> 📌 Make sure your first-frame label file is named as:  
> `TestLabels_FirstFrameOnly/<video_name>.txt`

---

## 📄 First Frame Label Format

First-frame labels should be in a `.txt` file matching the video name. Each line should include:

```txt
id class_id x1 y1 x2 y2 confidence
```

Example:
```
1 1 192.07 172.88 201.49 178.28 1.0
2 1 211.73 125.52 221.97 131.27 1.0
...
```

These are used to initialize object identities during tracking.

---

## 🧠 How It Works

1. **Detection** → YOLOv8 runs on each frame
2. **First-frame Matching** → On frame 1, detections are matched with labels to fix initial IDs
3. **Tracking** → StrongSORT maintains IDs using Kalman Filter + Deep ReID
4. **Logging & Saving** → Videos, track logs, and optionally bounding boxes are saved

---

## 🧪 Sample Output (Log)

```
video 1/1 (frame 1/750) MultiUAV-154.mp4: 512x640 18 DRONEs, 18.9ms
Warning: Skipping box with missing ID at frame 2
...
video 1/1 (frame 7/750) MultiUAV-154.mp4: 512x640 16 DRONEs, 12.6ms
```

Output results are saved to the `/outputs/` directory.

---

## 🧩 Customization Tips

- Modify `track.py` to:
  - Add new label parsing logic
  - Switch between `bytetrack` or `ocsort` trackers
- Use `--show-vid` or `--save-txt` for visual or coordinate outputs

---

## 🔍 Dependencies

- Python 3.10
- PyTorch >= 1.13.0
- OpenCV
- YOLOv8 (Ultralytics)
- DeepSORT / StrongSORT modules

Install all using:
```bash
pip install -r requirements.txt
```

---

## 🙌 Acknowledgments

- 📦 [BOXMot](https://github.com/mikel-brostrom/boxmot)
- 🤖 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- 🎓 [StrongSORT](https://github.com/dyhBUPT/StrongSORT)

---

## 📜 License

MIT License. Feel free to use and modify with credit.

---


