# 🎯 Object Detection with Voice Feedback for the Visually Impaired

This project is an AI-powered system built on Raspberry Pi to assist visually impaired individuals by detecting people and objects in front of them and providing **real-time voice feedback**. It uses a custom-trained YOLOv8 model along with a Python-based voice engine to convey detected objects through speech.

📘 **Based on the research paper:**  
[Obstacle Detection System for the Visually Impaired using Deep Learning and Audio Feedback](https://doi.org/10.1109/DICCT61038.2024.10533162)  
📄 DOI: 10.1109/DICCT61038.2024.10533162

---

## 🚀 Features

- 🧠 Object detection using **YOLOv8**
- 🗣️ **Voice feedback** using `pyttsx3`
- 🎥 Works with video files or real-time camera feeds
- ⚡ Lightweight and suitable for **Raspberry Pi**
- 🔧 Easy to modify and extend

---

## 🗂️ Project Structure

```
📁 .idea/              → Project settings (optional)
🎞️ aitvideo            → Sample video for demo
📄 gpModel.pt          → Custom YOLO model file
🐍 main.py             → Main Python script
📝 README.md           → Project documentation
🎞️ test_video*         → Test videos for detection
📄 yolo11n.pt          → YOLOv8n model file
📄 yolov8n.pt          → Pretrained YOLOv8n weights
```

---

## 🛠️ Tech Stack

| Component        | Technology        |
|------------------|-------------------|
| Model            | YOLOv8 (Ultralytics) |
| Language         | Python 3.x        |
| Vision Processing| OpenCV            |
| Voice Feedback   | pyttsx3 (offline TTS) |
| Hardware         | Raspberry Pi (tested on 4B) |

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/blind-aid-object-detector.git
   cd blind-aid-object-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install ultralytics opencv-python pyttsx3
   ```

3. Run the main script:
   ```bash
   python main.py
   ```

---

## 🧪 Sample Usage

- You can test with:
  ```bash
  python main.py --video test_video.mp4
  ```
- For live webcam (default):
  ```bash
  python main.py
  ```

Detected objects will be announced using audio feedback in real-time.

---

## 📚 Research Reference

This project was built following the concepts proposed in the IEEE paper:

> **Obstacle Detection System for the Visually Impaired using Deep Learning and Audio Feedback**  
> Presented at the DICCT Conference, 2024  
> DOI: [10.1109/DICCT61038.2024.10533162](https://doi.org/10.1109/DICCT61038.2024.10533162)
---

## 🤝 Contributions

Pull requests and suggestions are welcome!  
For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

This project is for **educational and research purposes** only.  
Please cite the original research paper if you use this work in any academic work.

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics) for YOLOv8
- The authors of the referenced IEEE research paper
- Open-source Python libraries and the Raspberry Pi community

---

## 📞 Contact

For any queries or feedback, feel free to raise an issue on the repository or connect via email.
