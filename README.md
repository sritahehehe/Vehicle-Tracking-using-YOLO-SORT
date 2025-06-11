# 🚦 Real-Time Vehicle Tracking using YOLOv8 and SORT

This project demonstrates real-time object detection and tracking using **YOLOv8** and **SORT**. Vehicles are detected using a pre-trained YOLO model, and each one is tracked across frames with a unique ID using the SORT tracking algorithm.

> 👥 Team Project by [Your Name] and [Friend’s Name]

---

## 📌 Features

- 🔍 Detects and tracks only **vehicles** (car, bus, truck, bike)
- 🧠 Combines YOLOv8 for detection with SORT for tracking
- 🎯 Assigns unique tracking IDs to each vehicle
- 🖼️ Displays bounding boxes and IDs on each frame
- 💾 Saves processed video output with annotations

---

## 🧰 Tech Stack / Tools Used

- **Python**  
- **YOLOv8** (Ultralytics) – object detection  
- **SORT** – object tracking  
- **OpenCV** – frame processing & drawing  
- **NumPy** – numerical operations  

---

## 🧠 System Architecture

```text
Input Video
    ↓
YOLOv8 Object Detection
    ↓
Vehicle Class Filter (car, bus, etc.)
    ↓
SORT Object Tracker (assigns ID to each vehicle)
    ↓
Annotated Output Frame
