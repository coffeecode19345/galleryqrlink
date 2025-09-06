import streamlit as st
import sqlite3
import io
from PIL import Image
import uuid
import mimetypes
import base64
import os
try:
    from pyzbar import pyzbar
    import cv2
    import numpy as np
    QR_CODE_AVAILABLE = True
except ImportError as e:
    st.warning("QR code reading is disabled due to missing dependencies. Install 'pyzbar' and 'opencv-python' and ensure 'libzbar0' is installed.")
    QR_CODE_AVAILABLE = False

DB_PATH = "qr_gallery.db"
MAX_FILE_SIZE_MB = 5

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

def read_qr_code(file):
    """Read QR code from an uploaded image file."""
    if not QR_CODE_AVAILABLE:
        return None
    try:
        file.seek(0)
        bytes_data = file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        qr_codes = pyzbar.decode(cv2_img)
        return qr_codes[0].data.decode('utf-8') if qr_codes else None
    except Exception as e:
        st.warning(f"Error reading QR code from {file.name}: {str(e)}")
        return None

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
            qr_content = read_qr_code(uploaded_file)
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
        if st.button("Upload Images", key="upload_button") and uploaded_files:
            uploaded_count = load_images_to_db(uploaded_files, folder_choice)
            st.success(f"{uploaded_count} image(s) uploaded to '{folder_choice}'!")
            if QR_CODE_AVAILABLE:
                for file in uploaded_files:
                    qr_content = read_qr_code(file)
                    if qr_content:
                        st.write(f"**{file.name} QR Content:** {qr_content}")
                    else:
                        st.write(f"**{file.name} QR Content:** No QR code detected")

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
    st.image(img_dict["image"], use_container_width=True)
    qr_display = img_dict["qr_content"] if img_dict["qr_content"] else "No QR code detected"
    st.markdown(f'<b>QR Content:</b> {qr_display}', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        if idx > 0 and st.button("◄ Previous", key=f"prev_{folder}_{idx}"):
            st.session_state.zoom_index -= 1
            st.rerun()
    with col3:
        if idx < len(images)-1 and st.button("Next ►", key=f"next_{folder}_{idx}"):
            st.session_state.zoom_index += 1
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
        st.rerun()
