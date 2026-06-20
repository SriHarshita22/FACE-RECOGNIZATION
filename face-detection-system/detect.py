import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Load Haar Cascade paths from OpenCV
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

# Initialize Haar Cascades
face_cascade = cv2.CascadeClassifier(face_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

def load_known_faces(registry_dir):
    """
    Loads all registered faces from registry_dir.
    Processes each image into a normalized 150x150 grayscale template.
    """
    known_faces = {}
    if not os.path.exists(registry_dir):
        os.makedirs(registry_dir)
        return known_faces
        
    for filename in os.listdir(registry_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(registry_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Resize and equalize to ensure illumination invariance
                    gray = cv2.resize(gray, (150, 150))
                    gray = cv2.equalizeHist(gray)
                    known_faces[name] = gray
                except Exception as e:
                    print(f"Error loading template {filename}: {e}")
    return known_faces

def register_new_face(face_img, name, registry_dir):
    """
    Saves a face crop into the face registry.
    """
    if not os.path.exists(registry_dir):
        os.makedirs(registry_dir)
        
    # Resize to standard size (200x200 BGR) and save
    face_resized = cv2.resize(face_img, (200, 200))
    # Standardize filename to prevent path issues
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
    filepath = os.path.join(registry_dir, f"{safe_name}.jpg")
    cv2.imwrite(filepath, face_resized)
    return filepath, safe_name

def recognize_face(face_img, known_faces_dict, threshold=0.4):
    """
    Recognizes a face image using normalized template matching.
    """
    if not known_faces_dict:
        return "Unknown", 0.0
        
    try:
        # Convert crop to grayscale, resize to 150x150, and equalize
        gray_crop = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray_crop = cv2.resize(gray_crop, (150, 150))
        gray_crop = cv2.equalizeHist(gray_crop)
        
        best_name = "Unknown"
        best_score = -1.0
        
        for name, template in known_faces_dict.items():
            # Perform correlation matching
            res = cv2.matchTemplate(gray_crop, template, cv2.TM_CCOEFF_NORMED)
            score = res[0][0]
            if score > best_score:
                best_score = score
                best_name = name
                
        if best_score >= threshold:
            return best_name, float(best_score)
            
        return "Unknown", float(best_score)
    except Exception as e:
        print(f"Error in face recognition: {e}")
        return "Unknown", 0.0

def detect_mask(face_img, skin_threshold=0.32):
    """
    Detects if a face is wearing a mask using skin color ratios in the lower half of the face.
    """
    try:
        h, w, _ = face_img.shape
        if h < 20 or w < 20:
            return "Unknown", 0.0
            
        # Crop the lower 55% of the face (mouth and nose region)
        lower_half = face_img[int(h*0.45):h, :]
        
        # Convert to YCrCb (better chrominance separation for skin detection)
        lower_ycrcb = cv2.cvtColor(lower_half, cv2.COLOR_BGR2YCrCb)
        
        # Skin color boundaries in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        skin_mask = cv2.inRange(lower_ycrcb, lower_skin, upper_skin)
        
        skin_pixels = np.sum(skin_mask == 255)
        total_pixels = skin_mask.size
        skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 1.0
        
        # If the skin ratio is low, a mask is likely worn
        is_masked = skin_ratio < skin_threshold
        
        return "Mask" if is_masked else "No Mask", float(skin_ratio)
    except Exception as e:
        print(f"Error in mask detection: {e}")
        return "Unknown", 0.0

def detect_emotion(face_img, face_gray):
    """
    Detects simple emotions: Happy, Surprised, Neutral.
    Uses OpenCV cascades and geometric eye heuristics.
    """
    try:
        h, w = face_gray.shape
        if h < 20 or w < 20:
            return "Neutral"
            
        # 1. Smile Detection (Happy)
        lower_half_gray = face_gray[int(h*0.55):h, :]
        
        # Detect smiles (scaleFactor=1.5, minNeighbors=18 for high precision)
        smiles = smile_cascade.detectMultiScale(lower_half_gray, scaleFactor=1.5, minNeighbors=18)
        if len(smiles) > 0:
            return "Happy"
            
        # 2. Eye Aspect Ratio (Surprised vs Neutral)
        upper_half_gray = face_gray[0:int(h*0.65), :]
        eyes = eye_cascade.detectMultiScale(upper_half_gray, scaleFactor=1.1, minNeighbors=4)
        
        if len(eyes) >= 2:
            # Sort by X coordinate to get left and right eye
            eyes = sorted(eyes, key=lambda x: x[0])
            # Average height relative to face height
            avg_eye_height = (eyes[0][3] + eyes[1][3]) / 2.0
            norm_eye_height = avg_eye_height / h
            
            # If eyes are wide open, class is Surprised
            if norm_eye_height > 0.16:
                return "Surprised"
                
        return "Neutral"
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return "Neutral"

def log_attendance(name, emotion, mask_status, attendance_file="data/attendance.csv"):
    """
    Saves attendance to a CSV file. Avoids duplicates of the same person within 15 seconds.
    """
    try:
        # Create directories if missing
        dir_name = os.path.dirname(attendance_file)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        
        # Initialize file if not existing
        if not os.path.exists(attendance_file):
            df = pd.DataFrame(columns=["Timestamp", "Date", "Name", "Emotion", "MaskStatus"])
            df.to_csv(attendance_file, index=False)
            
        # Duplicate detection (within 15 seconds)
        try:
            df_exist = pd.read_csv(attendance_file)
            if not df_exist.empty:
                # Filter by name
                person_records = df_exist[df_exist["Name"] == name]
                if not person_records.empty:
                    last_record = person_records.iloc[-1]
                    last_time = datetime.strptime(last_record["Timestamp"], "%Y-%m-%d %H:%M:%S")
                    if (now - last_time).total_seconds() < 15:
                        return False  # Cooldown active
        except Exception as e:
            print(f"Error reading attendance logs: {e}")
            
        # Append record
        new_row = {
            "Timestamp": timestamp_str,
            "Date": date_str,
            "Name": name,
            "Emotion": emotion,
            "MaskStatus": mask_status
        }
        df_new = pd.DataFrame([new_row])
        df_new.to_csv(attendance_file, mode='a', header=False, index=False)
        return True
    except Exception as e:
        print(f"Error logging attendance: {e}")
        return False
