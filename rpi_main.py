import pyttsx3
from threading import Thread
from queue import Queue
from ultralytics import YOLO
import cv2
import numpy as np
import time

import os

def speak(q):
    while True:
        if not q.empty():
            label, distance, position = q.get()
            rounded_distance = round(distance * 2) / 2  # Round to integer or in steps of 0.5
            rounded_distance_str = str(int(rounded_distance)) if rounded_distance.is_integer() else str(rounded_distance)
            
            text = f"{label} IS {rounded_distance_str} METERS ON {position}"
            os.system(f'espeak -s 150 "{text}"')
            
            # Clear the queue so we don't build up stale audio announcements
            with q.mutex:
                q.queue.clear()
        else:
            time.sleep(0.1)  

# ----------------- OPTIMIZATIONS FOR RASPBERRY PI ----------------- #

class VideoStream:
    """Threaded video stream to eliminate camera lag by getting only the latest frame."""
    def __init__(self, src=0, resolution=(640, 480)):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            time.sleep(0.01) # Small delay to avoid CPU pinning

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()

    def isOpened(self):
        return self.stream.isOpened()


class InferenceThread:
    """Threaded YOLO inference to keep the display completely smooth (high FPS) while RPi processes heavy math."""
    def __init__(self, model):
        self.model = model
        self.frame = None
        self.results = None
        self.boxes = []
        self.stopped = False
        self.new_frame = False

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if self.new_frame and self.frame is not None:
                # Lowering imgsz to 160 dramatically speeds up Raspberry Pi inference 
                results = self.model.predict(self.frame, imgsz=160, device='cpu', verbose=False)
                self.results = results[0]
                self.boxes = self.results.boxes
                self.new_frame = False
            else:
                time.sleep(0.01)

    def set_frame(self, frame):
        if not self.new_frame:
            self.frame = frame.copy()
            self.new_frame = True

    def get_results(self):
        return self.results, self.boxes

    def stop(self):
        self.stopped = True

# ------------------------------------------------------------------ #

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
    y1, y2 = max(0, y), min(image.shape[0], y+int(0.08 * h))
    x1, x2 = max(0, x), min(image.shape[1], x+w)
    
    top_region = image[y1:y2, x1:x2]
    if top_region.size > 0:
        blurred_top_region = cv2.GaussianBlur(top_region, (15, 15), 0)
        image[y1:y2, x1:x2] = blurred_top_region
    return image


def main():
    queue = Queue()
    t = Thread(target=speak, args=(queue,))
    t.daemon = True 
    t.start()

    model = YOLO("gpModel.pt")
    
    # Check if NCNN export exists. If yes, it's faster.
    # Otherwise, fallback to PyTorch pt file automatically.
    import os
    if os.path.exists("gpModel_ncnn_model"):
        print("Using super fast NCNN optimized model!")
        model = YOLO("gpModel_ncnn_model")
    else:
        print("Note: To optimize further on Raspberry Pi, you can export to NCNN:")
        print("      Run this command in terminal: yolo export format=ncnn model=gpModel.pt")

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

    print("Initializing Multi-Threaded Camera...")
    cap = VideoStream(0).start()
    # Add a small delay allowing camera to warm up
    time.sleep(1.0)
    
    print("Initializing Background YOLO Inference Thread...")
    inference_thread = InferenceThread(model).start()

    pause = False
    
    # Metrics
    fps_start_time = time.time()
    fps_frames = 0
    fps = 0.0

    while cap.isOpened():
        if not pause:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
                
            fps_frames += 1
            elapsed = time.time() - fps_start_time
            if elapsed > 1.0:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_start_time = time.time()
                
            # Send latest frame to background inference thread
            inference_thread.set_frame(frame)
            
            # Fetch latest available results
            result, boxes = inference_thread.get_results()

            nearest_object = None
            min_distance = float('inf')

            if result is not None and len(boxes) > 0:
                for box in boxes:
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

                    if label == "person":
                        frame = blur_person(frame, box)
                        cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorGreen, thickness)
                        cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorGreen, thickness)
                    elif label == "car":
                        cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorYellow, thickness)
                        cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorYellow, thickness)
                    elif label in class_avg_sizes:
                        cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorBlue, thickness)
                        cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorBlue, thickness)

                if nearest_object:
                    if nearest_object[0] in class_avg_sizes:  
                        cv2.rectangle(frame, (nearest_object[2][0], nearest_object[2][1]),(nearest_object[2][2], nearest_object[2][3]), (0, 0, 255), thickness)
                        text = f"{nearest_object[0]} - {round(nearest_object[1], 1)}m"
                        cv2.putText(frame, text, (nearest_object[2][0], nearest_object[2][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorRed, thickness)

                    if nearest_object[1] <= 12.5:  
                        position = get_position(frame.shape[1], nearest_object[2])
                        queue.put((nearest_object[0], nearest_object[1], position))

            # Display FPS visually
            cv2.putText(frame, f"Visual FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Audio World', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            pause = not pause

    cap.stop()
    inference_thread.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
