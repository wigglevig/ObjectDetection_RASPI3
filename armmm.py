import pyttsx3  # ensure `espeak` installed via `sudo apt install espeak`
from threading import Thread
from queue import Queue
from ultralytics import YOLO
import cv2
import numpy as np
import time

# Thread-safe queue for TTS messages
tts_queue = Queue()

# Preload average class sizes
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

# Initialize and run TTS engine in separate thread
def speak_loop(q: Queue):
    engine = pyttsx3.init(driverName='espeak')
    engine.setProperty('rate', 200)
    engine.setProperty('volume', 1.0)
    while True:
        if not q.empty():
            label, distance, position = q.get()
            # Round to nearest 0.5
            rounded = round(distance * 2) / 2
            text_dist = str(int(rounded)) if rounded.is_integer() else str(rounded)
            engine.say(f"{label} is {text_dist} meters to the {position}")
            engine.runAndWait()
            # clear any stale messages
            with q.mutex:
                q.queue.clear()
        else:
            time.sleep(0.1)

# Start TTS thread
tts_thread = Thread(target=speak_loop, args=(tts_queue,), daemon=True)
tts_thread.start()

# Helper functions
def calculate_distance(box, frame_width, names):
    x1, _, x2, _ = box.xyxy[0].cpu().numpy()
    pixel_width = x2 - x1
    label = names[int(box.cls[0])]
    if label in class_avg_sizes:
        pixel_width *= class_avg_sizes[label]['width_ratio']
    fov = 70  # degrees
    distance = (frame_width * 0.5) / np.tan(np.radians(fov / 2)) / (pixel_width + 1e-6)
    return float(distance)


def get_position(frame_width, box_coords):
    x1, _, x2, _ = box_coords
    cx = (x1 + x2) / 2
    if cx < frame_width / 3:
        return 'LEFT'
    elif cx < 2 * frame_width / 3:
        return 'FORWARD'
    else:
        return 'RIGHT'


def blur_person(image, box):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    h = y2 - y1
    top_h = int(0.1 * h)
    roi = image[y1:y1+top_h, x1:x2]
    if roi.size:
        image[y1:y1+top_h, x1:x2] = cv2.GaussianBlur(roi, (15,15), 0)
    return image

# Load a lightweight YOLO model
model = YOLO('gpModel.pt')
# Open video (or use 0 for Pi camera)
cap = cv2.VideoCapture(0)

frame = None
pause = False
while True:
    if not pause:
        ret, frame = cap.read()
        if not ret:
            break
        # reduce resolution for speed
        frame = cv2.resize(frame, (640, 480))
        # inference on CPU
        results = model.predict(frame, device='cpu', verbose=False)
        res = results[0]
        nearest = (None, float('inf'), None)

        for box in res.boxes:
            name = res.names[int(box.cls[0])]
            coords = [int(x) for x in box.xyxy[0].cpu().numpy()]
            d = calculate_distance(box, frame.shape[1], res.names)
            if d < nearest[1]:
                nearest = (name, d, coords)
            color = (255,0,0)
            if name == 'person':
                frame = blur_person(frame, box)
                color = (0,255,0)
            elif name == 'car':
                color = (0,255,255)
            if name in class_avg_sizes:
                cv2.rectangle(frame, (coords[0],coords[1]), (coords[2],coords[3]), color, 2)
                cv2.putText(frame, f"{name} {d:.1f}m", (coords[0], coords[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if nearest[0] and nearest[0] in class_avg_sizes:
            coords = nearest[2]
            cv2.rectangle(frame, (coords[0],coords[1]), (coords[2],coords[3]), (0,0,255), 2)
            cv2.putText(frame, f"{nearest[0]} {nearest[1]:.1f}m", (coords[0], coords[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            if nearest[1] < 12.5:
                pos = get_position(frame.shape[1], coords)
                tts_queue.put((nearest[0], nearest[1], pos))

    if frame is not None:
        cv2.imshow('RPI Audio World', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        pause = not pause

cap.release()
cv2.destroyAllWindows()
