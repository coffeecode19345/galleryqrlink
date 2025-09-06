import streamlit as st
import sqlite3
import io
from PIL import Image
import uuid
import mimetypes
import base64
import os
try:
    import cv2
    import numpy as np
    QR_CODE_AVAILABLE = True
except ImportError as e:
    st.error(f"QR code reading is disabled due to missing 'opencv-python': {str(e)}. Install 'opencv-python' to enable QR code detection.")
    QR_CODE_AVAILABLE = False
try:
    from ultralytics import YOLO
    ML_QR_DETECTION_AVAILABLE = True
except ImportError as e:
    st.warning(f"Machine learning QR detection is disabled: {str(e)}. Install 'ultralytics' for enhanced QR code region detection.")
    ML_QR_DETECTION_AVAILABLE = False

DB_PATH = "qr_gallery.db"
MAX_FILE_SIZE_MB = 5
MODEL_PATH = "yolo_qr.pt"  # Update to your fine-tuned model or use "yolov8n.pt"

# -------------------------------
# Helper Functions
# -------------------------------
def image_to_base64(image_data):
    """Convert image data (bytes) to base64 string."""
    return base64.b64encode(image_data).decode('utf-8') if isinstance(image_data, bytes) else image_data.encode('utf-8')

def validate_file(file):
    """Validate uploaded file size and type."""
    file_size_bytes = len(file.getvalue())
    if file_size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File '{file.name}' exceeds {MAX_FILE_SIZE_MB}MB limit.")
        return False
    file_type = file.type if hasattr(file, 'type') and file.type else os.path.splitext(file.name)[1].lower()
    if file_type not in ['image/jpeg', 'image/png', '.jpg', '.jpeg', '.png']:
        st.error(f"File '{file.name}' must be JPG or PNG.")
        return False
    try:
        file.seek(0)
        Image.open(file).verify()
        file.seek(0)
    except Exception as e:
        st.error(f"File '{file.name}' is invalid or corrupted: {str(e)}")
        return False
    return True

def preprocess_image(cv2_img):
    """Preprocess image to improve QR code detection."""
    try:
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return thresh
    except Exception as e:
        st.warning(f"Image preprocessing failed: {str(e)}")
        return cv2_img

def detect_qr_region(cv2_img):
    """Detect QR code region using YOLOv8."""
    if not ML_QR_DETECTION_AVAILABLE:
        return None
    try:
        model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else "yolov8n.pt"
        model = YOLO(model_path)
        results = model(cv2_img, classes=[0], conf=0.5)  # Adjust class ID and confidence
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            if len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0][:4])
                return (x1, y1, x2-x1, y2-y1)
        return None
    except Exception as e:
        st.warning(f"ML QR detection failed: {str(e)}")
        return None

def read_qr_code(file, zoom_region=None):
    """Read QR code from an uploaded image file using OpenCV's QRCodeDetector."""
    if not QR_CODE_AVAILABLE:
        return None, None
    try:
        file.seek(0)
        bytes_data = file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if cv2_img is None:
            st.warning(f"Failed to decode image {file.name}.")
            return None, None

        detector = cv2.QRCodeDetector()

        # Try ML-based QR region detection
        qr_region = detect_qr_region(cv2_img) if ML_QR_DETECTION_AVAILABLE else None
        if qr_region:
            x, y, w, h = qr_region
            crop = cv2_img[y:y+h, x:x+w]
            if crop.size > 0:
                qr_content, points = detector.detectAndDecode(crop)
                if qr_content:
                    return qr_content, crop
                crop_preprocessed = preprocess_image(crop)
                qr_content, points = detector.detectAndDecode(crop_preprocessed)
                if qr_content:
                    return qr_content, crop_preprocessed

        # Apply zoom region if provided
        if zoom_region:
            x, y, w, h = zoom_region
            crop = cv2_img[y:y+h, x:x+w]
            if crop.size > 0:
                qr_content, points = detector.detectAndDecode(crop)
                if qr_content:
                    return qr_content, crop
                crop_preprocessed = preprocess_image(crop)
                qr_content, points = detector.detectAndDecode(crop_preprocessed)
                if qr_content:
                    return qr_content, crop_preprocessed

        # Try original image
        qr_content, points = detector.detectAndDecode(cv2_img)
        if qr_content:
            return qr_content, cv2_img

        # Try preprocessed image
        preprocessed = preprocess_image(cv2_img)
        qr_content, points = detector.detectAndDecode(preprocessed)
        if qr_content:
            return qr_content, preprocessed

        # Try resizing
        for scale in [0.5, 1.5, 2.0]:
            resized = cv2.resize(cv2_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            qr_content, points = detector.detectAndDecode(resized)
            if qr_content:
                return qr_content, resized
            resized_preprocessed = preprocess_image(resized)
            qr_content, points = detector.detectAndDecode(resized_preprocessed)
            if qr_content:
                return qr_content, resized_preprocessed

        # Try cropping regions
        height, width = cv2_img.shape[:2]
        regions = [
            (0, 0, width//2, height//2),  # Top-left
            (width//2, 0, width, height//2),  # Top-right
            (0, height//2, width//2, height),  # Bottom-left
            (width//2, height//2, width, height),  # Bottom-right
        ]
        for (x, y, w, h) in regions:
            crop = cv2_img[y:h, x:w]
            if crop.size == 0:
                continue
            qr_content, points = detector.detectAndDecode(crop)
            if qr_content:
                return qr_content, crop
            crop_preprocessed = preprocess_image(crop)
            qr_content, points = detector.detectAndDecode(crop_preprocessed)
            if qr_content:
                return qr_content, crop_preprocessed

        return None, cv2_img
    except Exception as e:
        st.warning(f"Error reading QR code from {file.name}: {str(e)}")
        return None, None

def zoom_image(img, zoom_level, center_x, center_y):
    """Apply zoom to a PIL image and return the zoomed region."""
    width, height = img.size
    zoom_factor = max(1.0, zoom_level)
    new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)
    
    x0 = max(0, int(center_x * width - new_width // 2))
    y0 = max(0, int(center_y * height - new_height // 2))
    x1 = min(width, x0 + new_width)
    y1 = min(height, y0 + new_height)
    
    if x1 - x0 < new_width:
        x0 = max(0, x1 - new_width)
    if y1 - y0 < new_height:
        y0 = max(0, y1 - new_height)
    
    cropped = img.crop((x0, y0, x1, y1))
    zoomed = cropped.resize((width, height), Image.LANCZOS)
    return zoomed, (x0, y0, x1-x0, y1-y0)

def init_db():
    """Initialize SQLite database with folders and images tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            folder TEXT NOT NULL,
            image_data BLOB NOT NULL,
            qr_content TEXT,
            FOREIGN KEY(folder) REFERENCES folders(folder)
        )
    """)
    default_folders = [
        {"name": "General", "description": "General QR code images", "folder": "general"},
        {"name": "Marketing", "description": "Marketing campaign QR codes", "folder": "marketing"},
        {"name": "Events", "description": "Event-related QR codes", "folder": "events"},
    ]
    for folder_data in default_folders:
        c.execute("SELECT COUNT(*) FROM folders WHERE folder = ?", (folder_data["folder"],))
        if c.fetchone()[0] == 0:
            c.execute("""
                INSERT INTO folders (folder, name, description)
                VALUES (?, ?, ?)
            """, (folder_data["folder"], folder_data["name"], folder_data["description"]))
        else:
            c.execute("""
                UPDATE folders
                SET name = ?, description = ?
                WHERE folder = ?
            """, (folder_data["name"], folder_data["description"], folder_data["folder"]))
    conn.commit()
    conn.close()

def load_folders():
    """Load all folders from the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT folder, name, description FROM folders")
    folders = [{"folder": r[0], "name": r[1], "description": r[2]} for r in c.fetchall()]
    conn.close()
    return folders

def add_folder(folder, name, description):
    """Add a new folder to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO folders (folder, name, description)
            VALUES (?, ?, ?)
        """, (folder, name, description or ""))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        st.error(f"Folder '{folder}' already exists.")
        return False
    except Exception as e:
        st.error(f"Error adding folder: {str(e)}")
        return False

def load_images_to_db(uploaded_files, folder):
    """Load uploaded images to the database with QR code content."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    uploaded_count = 0
    for uploaded_file in uploaded_files:
        if validate_file(uploaded_file):
            image_data = uploaded_file.read()
            extension = os.path.splitext(uploaded_file.name)[1].lower()
            random_filename = f"{uuid.uuid4()}{extension}"
            qr_content, _ = read_qr_code(uploaded_file)
            c.execute("SELECT COUNT(*) FROM images WHERE folder = ? AND name = ?", (folder, random_filename))
            if c.fetchone()[0] == 0:
                c.execute("INSERT INTO images (name, folder, image_data, qr_content) VALUES (?, ?, ?, ?)",
                          (random_filename, folder, image_data, qr_content))
                uploaded_count += 1
    conn.commit()
    conn.close()
    return uploaded_count

def get_images(folder):
    """Retrieve images and QR content for a folder."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, image_data, qr_content FROM images WHERE folder = ?", (folder,))
    images = []
    for r in c.fetchall():
        name, data, qr_content = r
        try:
            img = Image.open(io.BytesIO(data))
            base64_image = image_to_base64(data)
            images.append({"name": name, "image": img, "data": data, "qr_content": qr_content, "base64": base64_image})
        except Exception as e:
            st.error(f"Error loading image {name}: {str(e)}")
    conn.close()
    return images

def delete_image(folder, name):
    """Delete an image from the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM images WHERE folder = ? AND name = ?", (folder, name))
    conn.commit()
    conn.close()

# -------------------------------
# Initialize DB & Session State
# -------------------------------
init_db()
if "zoom_folder" not in st.session_state:
    st.session_state.zoom_folder = None
if "zoom_index" not in st.session_state:
    st.session_state.zoom_index = 0
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "zoom_level" not in st.session_state:
    st.session_state.zoom_level = 1.0
if "zoom_center_x" not in st.session_state:
    st.session_state.zoom_center_x = 0.5
if "zoom_center_y" not in st.session_state:
    st.session_state.zoom_center_y = 0.5

# -------------------------------
# Sidebar: Admin Controls
# -------------------------------
with st.sidebar:
    st.title("Admin Login")
    with st.form(key="login_form"):
        pwd = st.text_input("Password", type="password", key="login_password")
        if st.form_submit_button("Login", key="login_button"):
            if pwd == "admin123":
                st.session_state.is_admin = True
                st.success("Logged in as admin!")
            else:
                st.error("Incorrect password")
    if st.session_state.is_admin and st.button("Logout", key="logout_button"):
        st.session_state.is_admin = False
        st.success("Logged out")
        st.rerun()

    if st.session_state.is_admin:
        st.subheader("Manage Folders & Images")
        # Add Folder
        with st.form(key="add_folder_form"):
            new_folder = st.text_input("Folder Name (e.g., 'newfolder')", key="new_folder_input")
            new_name = st.text_input("Display Name", key="new_name_input")
            new_description = st.text_area("Description (optional)", key="new_description_input")
            if st.form_submit_button("Add Folder", key="add_folder_button"):
                if new_folder and new_name:
                    if add_folder(new_folder.lower(), new_name, new_description):
                        st.success(f"Folder '{new_folder}' added successfully!")
                        st.rerun()
                else:
                    st.error("Folder Name and Display Name are required.")

        # Upload Images
        data = load_folders()
        folder_choice = st.selectbox("Select Folder", [item["folder"] for item in data], key="upload_folder_select")
        uploaded_files = st.file_uploader(
            "Upload QR Code Images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'], key="upload_files"
        )
        debug_preprocessed = st.checkbox("Show preprocessed images (Admin)", key="debug_preprocessed")
        if st.button("Upload Images", key="upload_button") and uploaded_files:
            uploaded_count = load_images_to_db(uploaded_files, folder_choice)
            st.success(f"{uploaded_count} image(s) uploaded to '{folder_choice}'!")
            if QR_CODE_AVAILABLE:
                for file in uploaded_files:
                    qr_content, processed_img = read_qr_code(file)
                    if qr_content:
                        st.write(f"**{file.name} QR Content:** {qr_content}")
                    else:
                        st.write(f"**{file.name} QR Content:** No QR code detected")
                    if debug_preprocessed and processed_img is not None and st.session_state.is_admin:
                        processed_pil = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                        st.image(processed_pil, caption=f"Preprocessed {file.name}", use_container_width=True)

# -------------------------------
# CSS Styling
# -------------------------------
st.markdown("""
<style>
.folder-card {background: #f9f9f9; border-radius: 8px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
.folder-header {font-size:1.5em; color:#333; margin-bottom:10px;}
.image-grid {display:flex; flex-wrap:wrap; gap:10px;}
img {border-radius:4px; max-width:100px; object-fit:cover;}
.qr-content {margin-top:10px; word-break:break-all; font-size:0.9em;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Main App UI
# -------------------------------
st.title("📸 QR Code Image Manager")

if not QR_CODE_AVAILABLE:
    st.error("QR code functionality is disabled. Install 'opencv-python' to enable QR code detection.")
if not ML_QR_DETECTION_AVAILABLE:
    st.warning("Machine learning QR detection is disabled. Install 'ultralytics' for enhanced detection.")

data = load_folders()
if st.session_state.zoom_folder is None:
    # Grid View
    if not data:
        st.info("No folders available. Admins can create folders in the sidebar.")
    for f in data:
        st.markdown(
            f'<div class="folder-card"><div class="folder-header">'
            f'{f["name"]} ({f["description"] or "No description"})</div>',
            unsafe_allow_html=True
        )
        images = get_images(f["folder"])
        if images:
            cols = st.columns(4)
            for idx, img_dict in enumerate(images):
                with cols[idx % 4]:
                    if st.button("🔍 View", key=f"view_{f['folder']}_{idx}"):
                        st.session_state.zoom_folder = f["folder"]
                        st.session_state.zoom_index = idx
                        st.session_state.zoom_level = 1.0
                        st.session_state.zoom_center_x = 0.5
                        st.session_state.zoom_center_y = 0.5
                        # Try ML-based QR region detection to initialize zoom
                        if ML_QR_DETECTION_AVAILABLE and QR_CODE_AVAILABLE:
                            file = io.BytesIO(img_dict["data"])
                            cv2_img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                            qr_region = detect_qr_region(cv2_img)
                            if qr_region:
                                x, y, w, h = qr_region
                                st.session_state.zoom_center_x = (x + w/2) / cv2_img.shape[1]
                                st.session_state.zoom_center_y = (y + h/2) / cv2_img.shape[0]
                                st.session_state.zoom_level = max(1.0, min(5.0, cv2_img.shape[1] / w))
                        st.rerun()
                    st.image(img_dict["image"], use_container_width=True, caption=f"Photo {idx+1}")
                    qr_display = img_dict["qr_content"] if img_dict["qr_content"] else "No QR code detected"
                    st.markdown(f'<div class="qr-content"><b>QR Content:</b> {qr_display}</div>', unsafe_allow_html=True)
                    if st.session_state.is_admin:
                        if st.button("🗑️ Delete", key=f"delete_grid_{f['folder']}_{img_dict['name']}"):
                            delete_image(f["folder"], img_dict["name"])
                            st.success("Image deleted.")
                            st.rerun()
        else:
            st.warning(f"No images in '{f['name']}'")

else:
    # Zoom View
    folder = st.session_state.zoom_folder
    images = get_images(folder)
    idx = st.session_state.zoom_index
    if idx >= len(images):
        idx = 0
        st.session_state.zoom_index = 0
    img_dict = images[idx]

    st.subheader(f"🔍 Viewing {folder} ({idx+1}/{len(images)})")
    
    # Image Magnifier Controls
    zoom_level = st.slider("Zoom Level", 1.0, 5.0, st.session_state.zoom_level, 0.1, key=f"zoom_level_{folder}_{idx}")
    center_x = st.slider("Horizontal Center", 0.0, 1.0, st.session_state.zoom_center_x, 0.01, key=f"center_x_{folder}_{idx}")
    center_y = st.slider("Vertical Center", 0.0, 1.0, st.session_state.zoom_center_y, 0.01, key=f"center_y_{folder}_{idx}")
    
    # Update session state
    st.session_state.zoom_level = zoom_level
    st.session_state.zoom_center_x = center_x
    st.session_state.zoom_center_y = center_y
    
    # Apply zoom
    zoomed_image, zoom_region = zoom_image(img_dict["image"], zoom_level, center_x, center_y)
    st.image(zoomed_image, use_container_width=True, caption="Zoomed Image")
    
    # Try QR code detection in zoomed region
    qr_display = img_dict["qr_content"] if img_dict["qr_content"] else "No QR code detected"
    if QR_CODE_AVAILABLE and zoom_level > 1.0:
        file = io.BytesIO(img_dict["data"])
        qr_content, processed_img = read_qr_code(file, zoom_region=(zoom_region[0], zoom_region[1], zoom_region[2], zoom_region[3]))
        if qr_content:
            qr_display = qr_content
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("UPDATE images SET qr_content = ? WHERE folder = ? AND name = ?",
                      (qr_content, folder, img_dict["name"]))
            conn.commit()
            conn.close()
        if st.session_state.is_admin and processed_img is not None:
            with st.expander("Show Preprocessed Zoomed Image"):
                processed_pil = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                st.image(processed_pil, caption="Preprocessed Zoomed Image")

    # Try ML-based QR detection if no QR code found
    if QR_CODE_AVAILABLE and ML_QR_DETECTION_AVAILABLE and not qr_display:
        file = io.BytesIO(img_dict["data"])
        cv2_img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        qr_region = detect_qr_region(cv2_img)
        if qr_region:
            file.seek(0)
            qr_content, processed_img = read_qr_code(file, zoom_region=qr_region)
            if qr_content:
                qr_display = qr_content
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("UPDATE images SET qr_content = ? WHERE folder = ? AND name = ?",
                          (qr_content, folder, img_dict["name"]))
                conn.commit()
                conn.close()
                x, y, w, h = qr_region
                st.session_state.zoom_center_x = (x + w/2) / cv2_img.shape[1]
                st.session_state.zoom_center_y = (y + h/2) / cv2_img.shape[0]
                st.session_state.zoom_level = max(1.0, min(5.0, cv2_img.shape[1] / w))
                st.rerun()

    st.markdown(f'<b>QR Content:</b> {qr_display}', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        if idx > 0 and st.button("◄ Previous", key=f"prev_{folder}_{idx}"):
            st.session_state.zoom_index -= 1
            st.session_state.zoom_level = 1.0
            st.session_state.zoom_center_x = 0.5
            st.session_state.zoom_center_y = 0.5
            st.rerun()
    with col3:
        if idx < len(images)-1 and st.button("Next ►", key=f"next_{folder}_{idx}"):
            st.session_state.zoom_index += 1
            st.session_state.zoom_level = 1.0
            st.session_state.zoom_center_x = 0.5
            st.session_state.zoom_center_y = 0.5
            st.rerun()

    mime, _ = mimetypes.guess_type(img_dict["name"])
    st.download_button(
        "⬇️ Download",
        data=img_dict["data"],
        file_name=f"{uuid.uuid4()}{os.path.splitext(img_dict['name'])[1]}",
        mime=mime,
        key=f"download_{folder}_{img_dict['name']}"
    )

    if st.session_state.is_admin:
        if st.button("🗑️ Delete Image", key=f"delete_zoom_{folder}_{img_dict['name']}"):
            delete_image(folder, img_dict["name"])
            st.success("Image deleted.")
            st.session_state.zoom_index = max(0, idx-1)
            if len(get_images(folder)) == 0:
                st.session_state.zoom_folder = None
                st.session_state.zoom_index = 0
            st.rerun()

    if st.button("⬅️ Back to Grid", key=f"back_{folder}_{idx}"):
        st.session_state.zoom_folder = None
        st.session_state.zoom_index = 0
        st.session_state.zoom_level = 1.0
        st.session_state.zoom_center_x = 0.5
        st.session_state.zoom_center_y = 0.5
        st.rerun()
