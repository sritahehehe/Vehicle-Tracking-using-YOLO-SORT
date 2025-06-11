import os
import cv2
from ultralytics import YOLO

# Fix OpenMP error on Windows (optional)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main(input_video_path=r"C:\Users\srita\yolo_sort_project\data\tress.mp4", 
         output_dir=r"C:\Users\srita\yolo_sort_project\outputs", show_gui=True):

    # Create output folder if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv8 model (make sure yolov8n.pt is downloaded)
    model = YOLO("yolov8n.pt")

    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # Setup video writer for output
    output_path = os.path.join(output_dir, "output.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {input_video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)[0]

        # Draw bounding boxes and labels
        for box in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{int(cls)} {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

        # Show frame if GUI enabled
        if show_gui:
            cv2.imshow("YOLO Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting early by user.")
                break

    cap.release()
    out.release()
    if show_gui:
        cv2.destroyAllWindows()

    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()
