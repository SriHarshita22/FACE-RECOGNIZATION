import streamlit as st  # type: ignore
import cv2  # type: ignore
import numpy as np
import os
import pandas as pd
import time
from PIL import Image
from datetime import datetime

# Configure page metadata and layout
st.set_page_config(
    page_title="VisionFace - AI Face Detection & Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local AI processing backend
from detect import (
    face_cascade,
    load_known_faces,
    register_new_face,
    recognize_face,
    detect_mask,
    detect_emotion,
    log_attendance
)

# Paths configuration
REGISTRY_DIR = "data/known_faces"
ATTENDANCE_FILE = "data/attendance.csv"
CAPTURED_DIR = "data/captured"

# Initialize directories
os.makedirs(REGISTRY_DIR, exist_ok=True)
os.makedirs(CAPTURED_DIR, exist_ok=True)

# Custom premium visual design CSS injection
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #080C14;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .main-title-container {
        padding: 1.5rem 0rem 0.5rem 0rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .main-title {
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 0.25rem;
    }
    
    /* Card Styles */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 1rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.3);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 0.25rem;
        background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Input adjustments */
    div[data-baseweb="select"] {
        border-radius: 0.5rem;
    }
    
    /* Info text */
    .info-label {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 0.2rem;
    }
    .info-value {
        font-size: 1rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 0.8rem;
    }
    
    /* Bordered section */
    .section-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Grid details */
    .face-detail-card {
        background: rgba(30, 41, 59, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Helper to load and prepare attendance data
def load_data():
    if not os.path.exists(ATTENDANCE_FILE):
        # Create folder and seed with sample data for display
        os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)
        now = datetime.now()
        dates = [now - pd.Timedelta(minutes=i*12) for i in range(5)]
        sample_records = [
            {"Timestamp": dates[4].strftime("%Y-%m-%d %H:%M:%S"), "Date": dates[4].strftime("%Y-%m-%d"), "Name": "Alice", "Emotion": "Happy", "MaskStatus": "No Mask"},
            {"Timestamp": dates[3].strftime("%Y-%m-%d %H:%M:%S"), "Date": dates[3].strftime("%Y-%m-%d"), "Name": "Bob", "Emotion": "Neutral", "MaskStatus": "Mask"},
            {"Timestamp": dates[2].strftime("%Y-%m-%d %H:%M:%S"), "Date": dates[2].strftime("%Y-%m-%d"), "Name": "Charlie", "Emotion": "Surprised", "MaskStatus": "No Mask"},
            {"Timestamp": dates[1].strftime("%Y-%m-%d %H:%M:%S"), "Date": dates[1].strftime("%Y-%m-%d"), "Name": "Alice", "Emotion": "Happy", "MaskStatus": "No Mask"},
            {"Timestamp": dates[0].strftime("%Y-%m-%d %H:%M:%S"), "Date": dates[0].strftime("%Y-%m-%d"), "Name": "Bob", "Emotion": "Neutral", "MaskStatus": "Mask"},
        ]
        df = pd.DataFrame(sample_records)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return df
    try:
        return pd.read_csv(ATTENDANCE_FILE)
    except:
        return pd.DataFrame(columns=["Timestamp", "Date", "Name", "Emotion", "MaskStatus"])

# Set up state managers
if 'registered_updated' not in st.session_state:
    st.session_state.registered_updated = False

# Load current known faces
known_faces = load_known_faces(REGISTRY_DIR)

# Sidebar layout
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h2 style='color: #6366F1; margin: 0; font-family: "Outfit"; font-weight: 800;'>VISIONFACE</h2>
    <p style='color: #64748B; font-size: 0.85rem; margin-top: 0.2rem;'>Computer Vision Dashboard</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation", 
    ["Dashboard", "Image Detection", "Webcam Live", "Face Registry", "Settings"],
    index=0
)

st.sidebar.markdown("---")

# Global detection thresholds in sidebar
st.sidebar.markdown("### Detection Settings")
rec_threshold = st.sidebar.slider("Recognition Match Threshold", 0.10, 1.00, 0.40, 0.05, 
                                  help="Min score required to match a registered face template.")
skin_threshold = st.sidebar.slider("Mask Skin Threshold", 0.10, 0.60, 0.32, 0.02, 
                                   help="Lower threshold classifes face as masked. Lower values = more mask sensitivity.")
scale_factor = st.sidebar.slider("Face Cascade Scale Factor", 1.05, 1.50, 1.10, 0.05,
                                 help="How much the image size is reduced at each image scale.")
min_neighbors = st.sidebar.slider("Face Cascade Min Neighbors", 1, 10, 5, 1,
                                  help="How many neighbors each candidate rectangle should have to retain it.")

# System details in sidebar
st.sidebar.markdown("---")
st.sidebar.caption("🤖 Powered by OpenCV Haar Cascades & Streamlit")
st.sidebar.caption(f"📅 Current Date: {datetime.now().strftime('%Y-%m-%d')}")

# Inject premium CSS style
inject_custom_css()

# Render Title Header
st.markdown(f"""
<div class="main-title-container">
    <h1 class="main-title">VisionFace: AI Face Analysis</h1>
    <div class="subtitle">{page} Page — Complete Facial Analysis, Recognition, and Attendance Tracking</div>
</div>
""", unsafe_allow_html=True)

# ----------------- PAGE 1: DASHBOARD -----------------
if page == "Dashboard":
    df = load_data()
    
    # Calculate stats
    total_detections = len(df)
    unique_faces = df["Name"].nunique() if total_detections > 0 else 0
    
    if total_detections > 0:
        mask_count = (df["MaskStatus"] == "Mask").sum()
        mask_compliance = int((mask_count / total_detections) * 100)
        
        happy_count = (df["Emotion"] == "Happy").sum()
        happy_pct = int((happy_count / total_detections) * 100)
    else:
        mask_compliance = 0
        happy_pct = 0
        
    # Render dashboard metrics in custom cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Detections</div>
            <div class="metric-value">{total_detections}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Registered People Detected</div>
            <div class="metric-value">{unique_faces}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mask Compliance</div>
            <div class="metric-value">{mask_compliance}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Happy Ratio</div>
            <div class="metric-value">{happy_pct}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### Analytics Charts")
    if total_detections > 0:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            st.markdown("#### Facial Emotions Distribution")
            # Aggregation for chart
            emotion_counts = df["Emotion"].value_counts().reset_index()
            emotion_counts.columns = ["Emotion", "Count"]
            st.bar_chart(data=emotion_counts, x="Emotion", y="Count", height=280)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with chart_col2:
            st.markdown("<div class='section-container'>", unsafe_allow_html=True)
            st.markdown("#### Mask Compliance Status")
            mask_counts = df["MaskStatus"].value_counts().reset_index()
            mask_counts.columns = ["Status", "Count"]
            st.bar_chart(data=mask_counts, x="Status", y="Count", height=280)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Attendance log table
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        st.markdown("#### Full Attendance Log")
        
        # Search & Filter
        search_query = st.text_input("🔍 Search Logs by Name", "")
        filtered_df = df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df["Name"].str.contains(search_query, case=False, na=False)]
            
        st.dataframe(
            filtered_df.sort_values(by="Timestamp", ascending=False),
            use_container_width=True,
            column_config={
                "Timestamp": "Detection Time",
                "Date": "Date",
                "Name": "Name / ID",
                "Emotion": "Detected Emotion",
                "MaskStatus": "Mask Wear Status"
            }
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Log as CSV",
            data=csv,
            file_name="face_attendance_log.csv",
            mime="text/csv"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No logs captured yet. Try uploading an image or activating the webcam!")

# ----------------- PAGE 2: IMAGE DETECTION -----------------
elif page == "Image Detection":
    st.markdown("### Process Local Image")
    uploaded_file = st.file_uploader("Upload an image containing faces (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Load image using PIL
        pil_image = Image.open(uploaded_file)
        # Convert to numpy/OpenCV BGR format
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        st.write("Processing image...")
        
        # Image conversion
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Run face detection
        faces = face_cascade.detectMultiScale(
            image_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        st.success(f"Detected {len(faces)} face(s) in image!")
        
        if len(faces) > 0:
            # We'll copy image to draw bounding boxes
            disp_image = image_bgr.copy()
            detected_face_data = []
            
            for idx, (x, y, w, h) in enumerate(faces):
                face_crop = image_bgr[y:y+h, x:x+w]
                face_gray_crop = image_gray[y:y+h, x:x+w]
                
                # Perform Face Recognition
                name, score = recognize_face(face_crop, known_faces, threshold=rec_threshold)
                
                # Perform Mask Detection
                mask_status, skin_val = detect_mask(face_crop, skin_threshold=skin_threshold)
                
                # Perform Emotion Detection
                emotion = detect_emotion(face_crop, face_gray_crop)
                
                # Assign name formatting
                name_display = name if name == "Unknown" else f"{name} ({int(score*100)}%)"
                
                # Log attendance dynamically
                log_attendance(name, emotion, mask_status, ATTENDANCE_FILE)
                
                # Store face properties for displaying in list below
                detected_face_data.append({
                    "id": idx,
                    "crop": face_crop,
                    "name": name,
                    "score": score,
                    "mask": mask_status,
                    "skin_val": skin_val,
                    "emotion": emotion,
                    "bbox": (x, y, w, h)
                })
                
                # Draw bounding box
                # Green for Mask, Red for No Mask
                box_color = (46, 204, 113) if mask_status == "Mask" else (231, 76, 60)
                
                # Box thickness
                cv2.rectangle(disp_image, (x, y), (x+w, y+h), box_color, 3)
                
                # Label overlay
                overlay_text = f"{name} | {emotion} | {mask_status}"
                cv2.putText(disp_image, overlay_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)
            
            # Display marked up image
            col_img, col_info = st.columns([2, 1])
            
            with col_img:
                st.markdown("<div class='section-container'>", unsafe_allow_html=True)
                st.markdown("#### Detected Faces Overlay")
                # Convert back to RGB for Streamlit displaying
                disp_image_rgb = cv2.cvtColor(disp_image, cv2.COLOR_BGR2RGB)
                st.image(disp_image_rgb, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_info:
                st.markdown("<div class='section-container'>", unsafe_allow_html=True)
                st.markdown("#### Detections Analysis")
                
                for item in detected_face_data:
                    st.markdown(f"<div class='face-detail-card'>", unsafe_allow_html=True)
                    
                    # Columns inside details card
                    fcol1, fcol2 = st.columns([1, 2])
                    with fcol1:
                        # Display small crop image
                        crop_rgb = cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB)
                        st.image(crop_rgb, width=90)
                    with fcol2:
                        st.markdown(f"<div class='info-label'>Person:</div><div class='info-value'>{item['name']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='info-label'>Emotion:</div><div class='info-value'>{item['emotion']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='info-label'>Mask:</div><div class='info-value'>{item['mask']}</div>", unsafe_allow_html=True)
                    
                    # Inline Face Registration form if Unknown
                    if item["name"] == "Unknown":
                        st.markdown("---")
                        st.caption("Identify this person to register them in the registry:")
                        reg_name = st.text_input("Enter Name", key=f"reg_name_{item['id']}")
                        if st.button("Register Face", key=f"reg_btn_{item['id']}"):
                            if reg_name.strip():
                                filepath, safe_name = register_new_face(item["crop"], reg_name, REGISTRY_DIR)
                                st.success(f"Registered {safe_name} successfully!")
                                time.sleep(1)
                                # Force rerun to reload registry dictionary
                                st.rerun()
                            else:
                                st.warning("Please enter a valid name.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No faces detected in the image. Try adjusting detection settings in the sidebar.")

# ----------------- PAGE 3: WEBCAM LIVE -----------------
elif page == "Webcam Live":
    st.markdown("### Choose Webcam Feed Mode")
    webcam_tab1, webcam_tab2 = st.tabs(["🎥 Live Video Stream", "📸 Snapshot Capture"])
    
    with webcam_tab1:
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        st.markdown("#### Real-Time Video Face Detection")
        st.caption("Start the live feed below. Detection bounding boxes and tracking stats are updated in real-time.")
        
        # Camera options selector for Windows and multiple webcams compatibility
        cc_col1, cc_col2 = st.columns(2)
        with cc_col1:
            cam_idx = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1,
                                      help="Select 0 for default. If default doesn't work, try index 1, 2, or 3.")
        with cc_col2:
            cam_backend = st.selectbox("Webcam API Backend", ["Default (Any)", "DirectShow (Windows CAP_DSHOW)", "MSMF (Windows CAP_MSMF)"], index=1,
                                       help="DirectShow is highly recommended on Windows to avoid webcam loading hangs.")
        
        run_feed = st.toggle("Start Camera Feed")
        
        # Frame placeholder
        st_frame = st.empty()
        # Summary placeholder
        st_info = st.empty()
        
        if run_feed:
            # Resolve backend flag
            backend_flag = cv2.CAP_ANY
            if cam_backend == "DirectShow (Windows CAP_DSHOW)":
                backend_flag = cv2.CAP_DSHOW
            elif cam_backend == "MSMF (Windows CAP_MSMF)":
                backend_flag = cv2.CAP_MSMF
                
            cap = cv2.VideoCapture(int(cam_idx), backend_flag)
            if not cap.isOpened():
                st.error("Error: Could not open local webcam. Verify your camera is connected and not in use by another app.")
                run_feed = False
            
            while run_feed:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera.")
                    break
                    
                # Mirror frame
                frame = cv2.flip(frame, 1)
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Face detection
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(50, 50)
                )
                
                # Face details tracking in current frame
                current_detections = []
                
                for idx, (x, y, w, h) in enumerate(faces):
                    face_crop = frame[y:y+h, x:x+w]
                    face_gray_crop = gray[y:y+h, x:x+w]
                    
                    # Inference
                    name, score = recognize_face(face_crop, known_faces, threshold=rec_threshold)
                    mask_status, _ = detect_mask(face_crop, skin_threshold=skin_threshold)
                    emotion = detect_emotion(face_crop, face_gray_crop)
                    
                    # Log attendance
                    log_attendance(name, emotion, mask_status, ATTENDANCE_FILE)
                    
                    current_detections.append(f"{name} ({emotion}, {mask_status})")
                    
                    # Box color (Green = Mask, Red = No Mask)
                    color = (46, 204, 113) if mask_status == "Mask" else (231, 76, 60)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Text overlays
                    label = f"{name} | {emotion} | {mask_status}"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, use_container_width=True)
                
                # Display info
                if len(faces) > 0:
                    detections_text = ", ".join(current_detections)
                    st_info.markdown(f"**Current Frame Detections ({len(faces)}):** {detections_text}")
                else:
                    st_info.markdown("*No faces detected in feed.*")
                
                # Small sleep to manage CPU usage
                time.sleep(0.03)
                
            if 'cap' in locals() and cap.isOpened():
                cap.release()
                
        st.markdown("</div>", unsafe_allow_html=True)
        
    with webcam_tab2:
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        st.markdown("#### Native Browser Camera Capture")
        st.caption("Capture a quick snapshot using your web browser's camera module. Ideal for remote testing.")
        
        camera_photo = st.camera_input("Take Snapshot")
        
        if camera_photo is not None:
            # Decode image
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            # Grayscale & Process
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                detected_face_data = []
                disp_image = img.copy()
                
                for idx, (x, y, w, h) in enumerate(faces):
                    face_crop = img[y:y+h, x:x+w]
                    face_gray_crop = gray[y:y+h, x:x+w]
                    
                    name, score = recognize_face(face_crop, known_faces, threshold=rec_threshold)
                    mask_status, skin_val = detect_mask(face_crop, skin_threshold=skin_threshold)
                    emotion = detect_emotion(face_crop, face_gray_crop)
                    
                    log_attendance(name, emotion, mask_status, ATTENDANCE_FILE)
                    
                    detected_face_data.append({
                        "id": idx,
                        "crop": face_crop,
                        "name": name,
                        "score": score,
                        "mask": mask_status,
                        "emotion": emotion
                    })
                    
                    # Bounding Box Color
                    color = (46, 204, 113) if mask_status == "Mask" else (231, 76, 60)
                    cv2.rectangle(disp_image, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(disp_image, f"{name} | {emotion}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                
                # Show snapshot output
                col_cap_img, col_cap_info = st.columns([2, 1])
                with col_cap_img:
                    st.image(cv2.cvtColor(disp_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col_cap_info:
                    st.markdown("#### Faces Details")
                    for item in detected_face_data:
                        st.markdown("<div class='face-detail-card'>", unsafe_allow_html=True)
                        ccol1, ccol2 = st.columns([1, 2])
                        with ccol1:
                            st.image(cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB), width=80)
                        with ccol2:
                            st.write(f"**Name:** {item['name']}")
                            st.write(f"**Emotion:** {item['emotion']}")
                            st.write(f"**Mask:** {item['mask']}")
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("No faces detected in the camera capture. Adjust parameters or try again.")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 4: FACE REGISTRY -----------------
elif page == "Face Registry":
    st.markdown("### Registered Known Faces Registry")
    
    # Reload registry
    known_faces = load_known_faces(REGISTRY_DIR)
    
    # Registry tabs
    reg_tab1, reg_tab2 = st.tabs(["👥 Registered Database", "➕ Register New Person"])
    
    with reg_tab1:
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        st.caption("Here is the list of all people currently recognized by the template matching face recognition backend.")
        
        if known_faces:
            # We display registered faces in columns (grid layout)
            files = [f for f in os.listdir(REGISTRY_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Show grid of 4 columns
            grid_cols = st.columns(4)
            for idx, filename in enumerate(files):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(REGISTRY_DIR, filename)
                col_idx = idx % 4
                
                with grid_cols[col_idx]:
                    st.markdown("<div class='face-detail-card' style='text-align: center;'>", unsafe_allow_html=True)
                    # Load crop
                    try:
                        registered_img = Image.open(img_path)
                        st.image(registered_img, width=130)
                    except:
                        st.error("Failed to load image.")
                    
                    st.markdown(f"**{name}**")
                    
                    # Delete button
                    if st.button("Delete Face", key=f"del_{name}"):
                        try:
                            os.remove(img_path)
                            st.success(f"Removed {name}!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {e}")
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No registered users found. Add faces under 'Register New Person' tab or from the Image Detection page!")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with reg_tab2:
        st.markdown("<div class='section-container'>", unsafe_allow_html=True)
        st.markdown("#### Upload & Enroll Face")
        st.caption("Provide a clean, well-lit frontal face image and enter their name. The system will detect their face and crop/save it as a template.")
        
        reg_input_name = st.text_input("Person's Full Name / ID")
        reg_upload = st.file_uploader("Upload Profile Photo", type=["jpg", "png", "jpeg"], key="reg_upload_main")
        
        if st.button("Enroll User"):
            if not reg_input_name.strip():
                st.warning("Please enter a name.")
            elif reg_upload is None:
                st.warning("Please upload an image.")
            else:
                # Load image
                pil_reg = Image.open(reg_upload)
                reg_bgr = cv2.cvtColor(np.array(pil_reg), cv2.COLOR_RGB2BGR)
                reg_gray = cv2.cvtColor(reg_bgr, cv2.COLOR_BGR2GRAY)
                
                # Detect face
                faces = face_cascade.detectMultiScale(reg_gray, 1.1, 5, minSize=(50,50))
                
                if len(faces) == 0:
                    st.error("Error: Face could not be detected in the uploaded profile image. Make sure it's a clear portrait.")
                elif len(faces) > 1:
                    st.warning("Warning: Multiple faces detected. Enrolling the largest face in image.")
                    # Sort faces by size, pick largest
                    largest_face = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
                    x, y, w, h = largest_face
                    face_crop = reg_bgr[y:y+h, x:x+w]
                    filepath, safe_name = register_new_face(face_crop, reg_input_name, REGISTRY_DIR)
                    st.success(f"Successfully registered {safe_name}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    x, y, w, h = faces[0]
                    face_crop = reg_bgr[y:y+h, x:x+w]
                    filepath, safe_name = register_new_face(face_crop, reg_input_name, REGISTRY_DIR)
                    st.success(f"Successfully registered {safe_name}!")
                    time.sleep(1)
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 5: SETTINGS -----------------
elif page == "Settings":
    st.markdown("### Database Operations & Reset Controls")
    
    st.markdown("<div class='section-container'>", unsafe_allow_html=True)
    st.markdown("#### Database Reset Controls")
    st.caption("WARNING: The actions below are permanent and cannot be undone.")
    
    col_reset1, col_reset2 = st.columns(2)
    
    with col_reset1:
        if st.button("🧹 Clear Attendance Log File"):
            if os.path.exists(ATTENDANCE_FILE):
                try:
                    os.remove(ATTENDANCE_FILE)
                    st.success("Successfully deleted attendance log database.")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error resetting logs: {e}")
            else:
                st.info("Log file is already empty.")
                
    with col_reset2:
        if st.button("🚨 Reset Whole Face Registry"):
            try:
                for f in os.listdir(REGISTRY_DIR):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        os.remove(os.path.join(REGISTRY_DIR, f))
                st.success("Successfully cleared all registered face templates.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting registry: {e}")
                
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-container'>", unsafe_allow_html=True)
    st.markdown("#### System Information & Environment Paths")
    
    st.write(f"**Attendance Database File Path:** `file:///{os.path.abspath(ATTENDANCE_FILE).replace(os.sep, '/')}`")
    st.write(f"**Known Faces Database Directory:** `file:///{os.path.abspath(REGISTRY_DIR).replace(os.sep, '/')}`")
    
    # Check dimensions of current logs
    if os.path.exists(ATTENDANCE_FILE):
        log_size = os.path.getsize(ATTENDANCE_FILE)
        st.write(f"**Logs Database Size:** {log_size} bytes")
    else:
        st.write("**Logs Database Size:** 0 bytes (empty)")
        
    st.markdown("</div>", unsafe_allow_html=True)
