import os
import cv2
from ultralytics import YOLO

# Fix OpenMP DLL error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main(input_video_path=r"C:\Users\srita\yolo_sort_project\data\tress.mp4", 
         output_dir=r"C:\Users\srita\yolo_sort_project\outputs", 
         show_gui=False):

    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv8 model (make sure yolov8n.pt is in the same folder)
    model = YOLO("yolov8n.pt")

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

    # Set up video writer
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

        # Draw bounding boxes
        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{int(cls)} {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the frame to output video
        out.write(frame)

        # Optionally show GUI
        if show_gui:
            cv2.imshow("YOLO Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quit key pressed. Exiting early.")
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Output saved at: {output_path}")

if __name__ == "__main__":
    main()
