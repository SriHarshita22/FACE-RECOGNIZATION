# VisionFace: AI Face Detection & Analysis System

Welcome to your advanced Computer Vision and Real-Time AI Attendance project!

VisionFace is a full-featured facial analysis dashboard. It uses Python, OpenCV, and Streamlit to detect human faces in images and live camera feeds, recognize individuals using a localized database of registered templates, analyze emotions, and detect mask compliance. All details are logged in an attendance database with visual analytics.

---

## 🎯 Key Features

1. **Robust Face Detection:**
   - Detects multiple faces in static image uploads and live webcam streams using OpenCV's Haar Cascade default model.
   - Configurable parameters (Scale Factor, Min Neighbors) for optimization in various lighting and density environments.

2. **Template-Based Face Recognition:**
   - Enrolls face profiles directly from static uploads or detection cropped regions.
   - Normalizes (150x150 grayscale & histogram equalization) face crops to compare correlation metrics.
   - Displays prediction labels with match confidence.

3. **Mask Compliance Detection:**
   - Evaluates the lower-face region (mouth and nose) for skin-tone distribution in YCrCb color space.
   - Triggers mask compliance status ("Mask" or "No Mask") with visual color-coded bounding boxes (Green = Mask, Red = No Mask).

4. **Smile & Eye Emotion Recognition:**
   - Detects "Happy" using the smile cascade.
   - Tracks eye openness (eye aspect metrics) to identify "Surprised" versus "Neutral" states.

5. **Visual Statistics Dashboard:**
   - Real-time aggregate metric counters.
   - Interactive charts for emotion breakdown and mask compliance.
   - Searchable, filterable log tables with CSV download capabilities.

---

## 📂 Project Structure

```
face-detection-system/
│
├── .streamlit/
│   └── config.toml        # Premium dark mode interface theme configuration
│
├── data/
│   ├── attendance.csv     # Attendance logs database
│   ├── known_faces/       # Registered face templates (enrollments)
│   └── captured/          # Saved face screenshots / crops
│
├── app.py                 # Multi-page Streamlit Dashboard app
├── detect.py              # AI inference, recognition, and heuristic engines
├── requirements.txt       # Dependencies
└── README.md              # Project Documentation
```

---

## 🚀 Installation & Quick Start

1. **Clone or Navigate to the Directory:**
   ```bash
   cd face-detection-system
   ```

2. **Install Dependencies:**
   Make sure you have Python 3.14+ installed. Run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard App:**
   ```bash
   streamlit run app.py
   ```
   Open your browser to the local URL (usually `http://localhost:8501`).

---

## ⚙️ How It Works (Heuristic Rationale)

- **Face Recognition:** Compares equalized grayscale crop metrics using OpenCV `matchTemplate` and normalized cross-correlation coefficient. This offers extremely low-latency sample matching without compiling heavy C++ packages like `dlib`.
- **Mask Detection:** Extracts the lower portion of the face. Standard skin tone in YCrCb chrominance space (Cr: 133–173, Cb: 77–127) is analyzed. If the skin pixel ratio drops below `0.32` (tunable in settings), the face is classified as wearing a mask.
- **Emotion Recognition:** Uses the Haar cascade smile model inside the lower face region and average eye bounding box aspect heights inside the upper face region.
