import pyttsx3
from threading import Thread
from queue import Queue
from ultralytics import YOLO
import cv2
import numpy as np
import time


def speak(q):
    # Fallback to default if espeak is somehow missing, but try espeak for Linux
    try:
        engine = pyttsx3.init(driverName='espeak')
    except:
        engine = pyttsx3.init()
        
    engine.setProperty('rate', 235)
    engine.setProperty('volume', 1.0)

    while True:
        if not q.empty():
            label, distance, position = q.get()
            rounded_distance = round(distance * 2) / 2  # Round to integer or in steps of 0.5
            # IF IT SAYS A INT NUMBER, IT REMOVES THE .0 PART. IT SAYS DIRECTLY 2 INSTEAD OF 2.0.
            rounded_distance_str = str(int(rounded_distance)) if rounded_distance.is_integer() else str(rounded_distance)
            
            # Note: class_avg_sizes is accessed from global scope, which is fine
            engine.say(f"{label} IS {rounded_distance_str} METERS ON {position}")
            engine.runAndWait()
            
            with queue.mutex:
                queue.queue.clear()
        else:
            time.sleep(0.1)  # To avoid busy waiting and to give delay
            
queue = Queue()
t = Thread(target=speak, args=(queue,))
t.daemon = True # Allows program to exit even if thread is running
t.start()

def calculate_distance(box, frame_width, class_avg_sizes, result):
    object_width = box.xyxy[0, 2].item() - box.xyxy[0, 0].item()

    label = result.names[box.cls[0].item()]

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
    # Adding max bounds so it doesn't crash if box is outside frame
    y1, y2 = max(0, y), min(image.shape[0], y+int(0.08 * h))
    x1, x2 = max(0, x), min(image.shape[1], x+w)
    
    top_region = image[y1:y2, x1:x2]
    if top_region.size > 0:
        blurred_top_region = cv2.GaussianBlur(top_region, (15, 15), 0)
        image[y1:y2, x1:x2] = blurred_top_region
    return image


model = YOLO("gpModel.pt")

# Set to 0 for camera, or "test_video.mp4" for video
cap = cv2.VideoCapture(0)  

# Add a small delay allowing camera to warm up
time.sleep(1.0)

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

pause = False
frame_count = 0
SKIP_FRAMES = 2  # skips 2 frames for every 1 processed to speed up FPS

# Variables to hold the last predictions so they stay on screen during skipped frames
last_result = None
last_boxes = []

while cap.isOpened():
    if not pause:
        ret, frame = cap.read()
        
        # FIX: The camera might occasionally fail to grab a frame. 
        # If we don't check 'ret', model.predict(frame) crashes!
        if not ret or frame is None:
            print("Failed to grab frame. Retrying...")
            continue
            
        frame_count += 1

        # We only run prediction every (SKIP_FRAMES + 1) frames
        if frame_count % (SKIP_FRAMES + 1) == 1:
            # imgsz=320 is the magic trick for RPi. It runs YOLO at low res internally 
            # while keeping your visual frame high res! device='cpu' prevents PyTorch bugs.
            results = model.predict(frame, imgsz=320, device='cpu', verbose=False)
            last_result = results[0]
            last_boxes = last_result.boxes

        result = last_result
        if result is None:
            continue

        nearest_object = None
        min_distance = float('inf')
        detected_objects = []

        for box in last_boxes:
            label = result.names[box.cls[0].item()]
            cords = [round(x) for x in box.xyxy[0].tolist()]
            colorGreen = (0, 255, 0)
            colorYellow = (0, 255, 255)
            colorBlue = (255, 0, 0)
            colorRed = (0, 0, 255)

            thickness = 2

            distance = calculate_distance(box, frame.shape[1], class_avg_sizes, result)

            if distance < min_distance:
                min_distance = distance
                nearest_object = (label, round(distance, 1), cords)
                detected_objects = [(label, round(distance, 1))]

            if label == "person":
                frame = blur_person(frame, box)
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

        if nearest_object:
            if nearest_object[0] in class_avg_sizes:  
                cv2.rectangle(frame, (nearest_object[2][0], nearest_object[2][1]),(nearest_object[2][2], nearest_object[2][3]), (0, 0, 255), thickness)
                text = f"{nearest_object[0]} - {round(nearest_object[1], 1)}m"
                cv2.putText(frame, text, (nearest_object[2][0], nearest_object[2][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colorRed, thickness)

            if nearest_object[1] <= 12.5:  
                position = get_position(frame.shape[1], nearest_object[2])
                queue.put((nearest_object[0], nearest_object[1], position))

            detected_objects.clear()
            
    else: # If paused
        if cap.retrieve()[0]:
            frame = cap.retrieve()[1]

    cv2.imshow('Audio World', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        pause = not pause

cap.release()
cv2.destroyAllWindows()
