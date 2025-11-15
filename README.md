# Human-Emotion-Detection-with-OpenCV

## Overview
This project detects **human emotions** from facial expressions using a **hybrid Computer Vision + Machine Learning** approach.  
We use **OpenCV**, **dlib** , **DeepFace**, and **tensorflow** for facial feature detection, and extracted geometric features are later classified using traditional ML models.

For all methods to work, use Python 3.10 or below

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/amaanaama/Human-Emotion-Detection-with-OpenCV.git
```

### 2. Create a virtual environment
```bash
python3.10 -m venv .venv
```

### 3. Activate the virtual environment
- **Linux / macOS**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell)**
  ```bash
  .venv\Scripts\activate
  ```

You should now see `(.venv)` appear before your terminal prompt.

---

### 4. Install dependencies
```bash
pip install opencv-python numpy matplotlib imutils deepface tensorflow dlib
```

---


---

## Reactivating the Environment

Each time you reopen the project:
```bash
cd ~/Documents/Code/CV-human-facial-detection
source .venv/bin/activate
```

When you’re done:
```bash
deactivate
```

---
