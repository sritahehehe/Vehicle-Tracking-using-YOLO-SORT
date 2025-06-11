import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

# Fix OpenMP DLL error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define class IDs of vehicles in COCO dataset
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

def main(input_video_path=r"C:\Users\srita\yolo_sort_project\data\tress.mp4", 
         output_dir=r"C:\Users\srita\yolo_sort_project\outputs", 
         show_gui=False):

    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Initialize SORT tracker
    tracker = Sort()

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video loaded: {input_video_path}")
    print(f"[INFO] Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

    output_path = os.path.join(output_dir, "output.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or read error.")
            break

        frame_num += 1
        print(f"[INFO] Processing frame {frame_num}/{total_frames}...")

        # Run YOLOv8 detection
        results = model(frame)[0]

        detections = []
        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            if int(cls) in VEHICLE_CLASSES:
                detections.append([x1, y1, x2, y2, conf])

        tracked_objects = tracker.update(np.array(detections))

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Output saved at: {output_path}")

if __name__ == "__main__":
    main()
