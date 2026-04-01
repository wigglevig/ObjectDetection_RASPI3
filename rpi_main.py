import pyttsx3
from threading import Thread
from queue import Queue
from ultralytics import YOLO
import cv2
import numpy as np
import time

# --- CONFIGURATION ---
USE_CAMERA = True          # Set to True to use live camera, False to use test_video.mp4
SKIP_FRAMES = 2            # Run object detection every N frames to improve performance on Pi
INFERENCE_SIZE = 320       # Smaller size for faster processing, keeps original video output size
CAMERA_INDEX = 0           # The camera port (0 is default for Pi/USB cameras)
# ---------------------

def speak(q):
    # Depending on the system, the driver might vary. 'espeak' is used in Linux.
    try:
        engine = pyttsx3.init(driverName='espeak')
    except Exception:
        engine = pyttsx3.init() # fallback to default

    engine.setProperty('rate', 235)
    engine.setProperty('volume', 1.0)

    while True:
        if not q.empty():
            label, distance, position = q.get()
            rounded_distance = round(distance * 2) / 2  # Round to integer or in steps of 0.5
            # IF IT SAYS A INT NUMBER, IT REMOVES THE .0 PART. IT SAYS DIRECTLY 2 INSTEAD OF 2.0.
            rounded_distance_str = str(int(rounded_distance)) if rounded_distance.is_integer() else str(rounded_distance)
            
            # Using global class_avg_sizes requires it to be defined. We will pass it via queue technically,
            # but reading global is fine here as it's static.
            engine.say(f"{label} IS {rounded_distance_str} METERS ON {position}")
            engine.runAndWait()
            
            # Clear queue to avoid playing backlog of warnings
            with q.mutex:
                q.queue.clear()
        else:
            time.sleep(0.1)  # To avoid busy waiting and to give delay

queue = Queue()
t = Thread(target=speak, args=(queue,), daemon=True)
t.start()

def calculate_distance(box, frame_width, class_avg_sizes, label):
    object_width = box.xyxy[0, 2].item() - box.xyxy[0, 0].item()

    if label in class_avg_sizes:
        object_width *= class_avg_sizes[label]["width_ratio"]

    distance = (frame_width * 0.5) / np.tan(np.radians(70 / 2)) / (object_width + 1e-6)
    return round(distance, 2)


def get_position(frame_width, box):
    if box[0] < frame_width // 3:
        return "LEFT"
    elif box[0] < 2 * (frame_width // 3):
        return "FORWARD"
    else:
        return "RIGHT"


def blur_person(image, box):
    x, y, w, h = box.xyxy[0].cpu().numpy().astype(int)
    # Ensure coordinates are within image bounds
    y1, y2 = max(0, y), min(image.shape[0], y + int(0.08 * h))
    x1, x2 = max(0, x), min(image.shape[1], x + w)
    
    top_region = image[y1:y2, x1:x2]
    if top_region.size > 0:
        blurred_top_region = cv2.GaussianBlur(top_region, (15, 15), 0)
        image[y1:y2, x1:x2] = blurred_top_region
    return image


class_avg_sizes = {
    "person": {"width_ratio": 2.5},
    "car": {"width_ratio": 0.37},
    "bicycle": {"width_ratio": 2.3},
    "motorcycle": {"width_ratio": 2.4},
    "bus": {"width_ratio": 0.3},
    "traffic light": {"width_ratio": 2.95},
    "stop sign": {"width_ratio": 2.55},
    "bench": {"width_ratio": 1.6},
    "cat": {"width_ratio": 1.9},
    "dog": {"width_ratio": 1.5},
}


def main():
    print("Loading YOLO model...")
    try:
        model = YOLO("gpModel.pt")
    except Exception as e:
        print(f"Error loading gpModel.pt, falling back to yolov8n.pt: {e}")
        model = YOLO("yolov8n.pt")

    if USE_CAMERA:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    else:
        cap = cv2.VideoCapture("test_video.mp4")

    if not cap.isOpened():
        print("Error: Could not open camera or video file.")
        return

    pause = False
    frame_count = 0

    # Variables to hold previous results for drawing during skipped frames
    prev_results = []
    prev_nearest = None

    while cap.isOpened():
        if not pause:
            ret, frame = cap.read()
            if not ret:
                if not USE_CAMERA:
                    # Restart video if using file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            
            frame_count += 1
            
            # Predict only every logic tick
            if frame_count % (SKIP_FRAMES + 1) == 0 or frame_count == 1:
                # Resize specifically for YOLO inference speed on Pi
                small_frame = cv2.resize(frame, (INFERENCE_SIZE, INFERENCE_SIZE))
                results = model.predict(small_frame, device='cpu', verbose=False)
                result = results[0]
                
                nearest_object = None
                min_distance = float('inf')
                current_results = []

                # Calculate scaling factors
                scale_x = frame.shape[1] / INFERENCE_SIZE
                scale_y = frame.shape[0] / INFERENCE_SIZE

                for box in result.boxes:
                    label = result.names[box.cls[0].item()]
                    # Scale coordinates back up
                    raw_cords = box.xyxy[0].cpu().numpy()
                    cords = [
                        int(raw_cords[0] * scale_x),
                        int(raw_cords[1] * scale_y),
                        int(raw_cords[2] * scale_x),
                        int(raw_cords[3] * scale_y)
                    ]
                    
                    # We pass the scaled box width directly to distance calc
                    object_width = cords[2] - cords[0]
                    distance_width = object_width
                    if label in class_avg_sizes:
                        distance_width *= class_avg_sizes[label]["width_ratio"]
                    distance = (frame.shape[1] * 0.5) / np.tan(np.radians(70 / 2)) / (distance_width + 1e-6)
                    distance = round(distance, 2)

                    if distance < min_distance:
                        min_distance = distance
                        nearest_object = (label, round(distance, 1), cords)

                    current_results.append({
                        "label": label,
                        "cords": cords,
                        "distance": distance,
                        # Pass fake box object to blur_person to maintain function signature
                        "box": type('obj', (object,), {'xyxy': np.array([[cords[0], cords[1], cords[2], cords[3]]])})()
                    })

                prev_results = current_results
                prev_nearest = nearest_object
                
                if prev_nearest and prev_nearest[0] in class_avg_sizes:
                    if prev_nearest[1] <= 12.5:  
                        position = get_position(frame.shape[1], prev_nearest[2])
                        queue.put((prev_nearest[0], prev_nearest[1], position))
                        
            # Draw persistent results onto current frame
            for res in prev_results:
                label = res["label"]
                cords = res["cords"]
                distance = res["distance"]
                
                colorGreen = (0, 255, 0)
                colorYellow = (0, 255, 255)
                colorBlue = (255, 0, 0)
                thickness = 2
                
                if label == "person":
                    frame = blur_person(frame, res["box"])
                    cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorGreen, thickness)
                    cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, colorGreen, thickness)
                elif label == "car":
                    cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorYellow, thickness)
                    cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, colorYellow, thickness)
                elif label in class_avg_sizes:
                    cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorBlue, thickness)
                    cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, colorBlue, thickness)

            if prev_nearest and prev_nearest[0] in class_avg_sizes:
                cv2.rectangle(frame, (prev_nearest[2][0], prev_nearest[2][1]),(prev_nearest[2][2], prev_nearest[2][3]), (0, 0, 255), 2)
                text = f"{prev_nearest[0]} - {round(prev_nearest[1], 1)}m"
                cv2.putText(frame, text, (prev_nearest[2][0], prev_nearest[2][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

        cv2.imshow('Audio World', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            pause = not pause

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
