from email.mime import image
import os
import sqlite3
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, session, url_for, send_file, flash
from tensorflow.keras.models import load_model
import uuid
from io import BytesIO, StringIO
import csv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import Response
from werkzeug.security import generate_password_hash, check_password_hash

from services.mri_loader import load_mri
from services.slice_selector import find_best_slice
from services.segmentation_service import segment
from services.tumor_cropper import crop_tumor
from services.classification_service import classify
from services.stage_estimator import estimate_stage
from services.visualization_service import generate_overlay
from services.report_generator import generate_clinical_report


# =========================
# APP CONFIG
# =========================

app = Flask(__name__)
app.secret_key = "brain_tumor_project"

IMG_SIZE = 128

RESULT_FOLDER = "static"
MODEL_FOLDER = "model"

os.makedirs(RESULT_FOLDER, exist_ok=True)


# =========================
# DATABASE
# =========================

def init_db():

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS reports(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT,
        doctor_email TEXT,
        result TEXT,
        tumor_type TEXT,
        stage TEXT,
        confidence REAL DEFAULT 0.0,
        image_path TEXT,
        slice_index INTEGER DEFAULT 0,
        process_time REAL DEFAULT 0.0,
        status TEXT DEFAULT 'pending',
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()


# =========================
# LOSS FUNCTIONS
# =========================

def dice_loss(y_true, y_pred):

    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    return 1 - (2 * intersection + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):

    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


custom = {
    "dice_loss": dice_loss,
    "combined_loss": combined_loss
}


# =========================
# LOAD CLASSIFIER FIRST
# =========================

try:
    print("Loading classifier...")
    classifier_model = load_model(os.path.join(MODEL_FOLDER, "tumor_type_model.h5"), compile=False)
except Exception as e:
    print("OOM: Failed to load classifier model:", e)
    classifier_model = None

classes = ["glioma", "meningioma", "pituitary"]


# =========================
# LOAD SEGMENTATION MODELS
# =========================

def load_safe_model(path):
    if os.path.exists(path):
        try:
            print("Loading:", path)
            return load_model(path, custom_objects=custom, compile=False)
        except Exception as e:
            print(f"OOM: Skipping {path} due to memory limits: {e}")
            import gc
            gc.collect()
            return None
    print("Missing model:", path)
    return None

unet_model = load_safe_model(os.path.join(MODEL_FOLDER, "x1.h5"))
resunet_model = load_safe_model(os.path.join(MODEL_FOLDER, "y1.h5"))
attention_model = load_safe_model(os.path.join(MODEL_FOLDER, "z1.h5"))


# =========================
# HELPER FUNCTIONS
# =========================

def get_side_effects(tumor_type, stage):
    effects = {
        "glioma": {
            "Stage 1": "Headaches, mild cognitive issues, occasional seizures.",
            "Stage 2": "Frequent headaches, mild weakness on one side, memory issues, seizures.",
            "Stage 3": "Worsening headaches, cognitive decline, personality changes, weakness, visual disturbances.",
            "Stage 4": "Severe headaches, nausea, significant cognitive/personality changes, difficulty speaking, vision loss."
        },
        "meningioma": {
            "Stage 1": "Often asymptomatic. Possible mild headaches, focal seizures, or visual changes.",
            "Stage 2": "Headaches, weakness, memory loss, partial vision loss, seizures.",
            "Stage 3": "Significant neurological deficits, severe headaches, weakness, behavioral changes.",
            "Stage 4": "Severe neurological deficits, increased intracranial pressure, profound weakness and vision loss."
        },
        "pituitary": {
            "Stage 1": "Hormonal imbalances, mild headaches, mild visual field defects.",
            "Stage 2": "Significant vision loss, severe headaches, profound hormonal changes, fatigue, weight changes.",
            "Stage 3": "Severe headaches, cranial nerve palsies, major vision loss, hypopituitarism.",
            "Stage 4": "Intense headaches, severe visual and endocrine complications."
        }
    }
    
    if tumor_type and stage:
        t_type = tumor_type.lower()
        if t_type in effects and stage in effects[t_type]:
            return effects[t_type][stage]
            
    return "No significant clinical side effects or healthy scan."

def normalize(img):

    mean = np.mean(img)
    std = np.std(img)

    if std != 0:
        img = (img - mean) / std

    return img





# =========================
# LOGIN
# =========================

@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()

        cur.execute(
            "SELECT role, password FROM users WHERE email=?",
            (email,)
        )

        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session["user"] = email
            role = user[0]
            session["role"] = role
            
            # Store city for hospital recommendations (patients only)
            city = request.form.get("city", "").strip()
            if city:
                session["city"] = city

            if role == "doctor":
                return redirect("/doctor_dashboard")
            else:
                return redirect("/patient_dashboard")

        return render_template("login.html", error="Invalid Login")

    return render_template("login.html")


# =========================
# REGISTER
# =========================

@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        role = request.form["role"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()

        try:
            hashed_pw = generate_password_hash(password)
            cur.execute(
                "INSERT INTO users(name,email,password,role) VALUES(?,?,?,?)",
                (name,email,hashed_pw,role)
            )

            conn.commit()

        except:
            return render_template("register.html", error="User already exists")

        conn.close()

        return redirect("/login")

    return render_template("register.html")


# =========================
# LOGOUT
# =========================

@app.route("/logout")
def logout():

    session.pop("user",None)
    return redirect("/login")


# =========================
# PROFILE
# =========================

@app.route("/profile")
def profile():

    if "user" not in session:
        return redirect("/login")

    email = session["user"]

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    cur.execute("SELECT name, email FROM users WHERE email=?", (email,))
    user_record = cur.fetchone()
    conn.close()

    if user_record:
        user_name, user_email = user_record
    else:
        user_name, user_email = "Unknown", email

    city = session.get("city", "")
    role = session.get("role", "")
    return render_template("profile.html", name=user_name, email=user_email, city=city, role=role)


@app.route("/update_name", methods=["POST"])
def update_name():
    if "user" not in session:
        return redirect("/login")
        
    new_name = request.form["name"]
    email = session["user"]
    
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    
    # Get old name first
    cur.execute("SELECT name FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    old_name = row[0] if row else None
    
    cur.execute("UPDATE users SET name = ? WHERE email = ?", (new_name, email))
    
    if old_name:
        cur.execute("UPDATE reports SET patient_name = ? WHERE patient_name = ?", (new_name, old_name))
        
    conn.commit()
    conn.close()
    
    flash("Name updated successfully!", "success")
    return redirect("/profile")


@app.route("/update_city", methods=["POST"])
def update_city():
    if "user" not in session:
        return redirect("/login")
    city = request.form.get("city", "").strip()
    session["city"] = city
    flash("City updated successfully!", "success")
    return redirect("/profile")


@app.route("/hospitals")
def hospitals():
    """Render hospital search page for patients only."""
    if "user" not in session:
        return redirect("/login")
    if session.get("role") == "doctor":
        return redirect("/doctor_dashboard")
    city = session.get("city", "")
    return render_template("hospitals.html", city=city)

# =========================
# ANALYSIS MODES
# =========================

@app.route("/detect", methods=["GET", "POST"])
def detect_page():
    if "user" not in session:
        return redirect("/login")
    if session.get("role") != "doctor":
        return redirect("/patient_dashboard")
    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return render_template("detect.html", error="Upload MRI file")
        if not file.filename.endswith('.nii'):
            return render_template("detect.html", error="Invalid file type. Only .nii files are allowed.")

        import tempfile
        import time
        start_time = time.time()
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            file.save(tmp.name)
            filepath = tmp.name

        try:
            volume, header = load_mri(filepath)
        except:
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template("detect.html", error="Invalid MRI (.nii) file")

        try:
            import gc
            gc.collect()
            # Do NOT remove filepath here; we need it for run_full_analysis later
        except:
            pass

        slice_index = find_best_slice(volume)
        image = volume[:,:,slice_index]
        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image_norm = normalize(image_resized)
        input_img = image_norm[np.newaxis,...,np.newaxis]

        mask = segment(input_img, unet_model, resunet_model, attention_model)
        tumor_crop = crop_tumor(image_resized, mask)

        tumor_pixels = np.sum(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        tumor_ratio = tumor_pixels / total_pixels

        if tumor_ratio > 0.01:
            detection = "Tumor Detected"
            tumor_type, confidence = classify(tumor_crop, classifier_model, classes)
        else:
            detection = "No Tumor Detected"
            tumor_type = "None"
            confidence = 0.0

        patient_name = request.form.get("patient_name", "Unknown Patient")
        doctor_email = session.get("user")
        date_str = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        process_time = time.time() - start_time
        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO reports (patient_name, doctor_email, result, tumor_type, stage, confidence, image_path, slice_index, process_time, status, date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'temporary', ?)",
            (patient_name, doctor_email, detection, tumor_type, "N/A", confidence, filepath, slice_index, round(process_time, 2), date_str)
        )
        report_id = cur.lastrowid
        conn.commit()
        conn.close()

        return render_template(
            "detect.html",
            result=detection,
            tumor_type=tumor_type,
            confidence=round(confidence,2),
            process_time=round(process_time, 2),
            scan_id=report_id
        )
    return render_template("detect.html")

@app.route("/segment", methods=["GET", "POST"])
def segment_page():
    if "user" not in session:
        return redirect("/login")
    if session.get("role") != "doctor":
        return redirect("/patient_dashboard")
    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return render_template("segment.html", error="Upload MRI file")
        if not file.filename.endswith('.nii'):
            return render_template("segment.html", error="Invalid file type. Only .nii files are allowed.")

        import tempfile
        import time
        start_time = time.time()
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            file.save(tmp.name)
            filepath = tmp.name

        try:
            volume, header = load_mri(filepath)
        except:
            if os.path.exists(filepath):
                os.remove(filepath)
            return render_template("segment.html", error="Invalid MRI (.nii) file")

        try:
            import gc
            gc.collect()
            # Do NOT remove filepath here; we need it for run_full_analysis later
        except:
            pass

        slice_index = find_best_slice(volume)
        image = volume[:,:,slice_index]
        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image_norm = normalize(image_resized)
        input_img = image_norm[np.newaxis,...,np.newaxis]

        mask = segment(input_img, unet_model, resunet_model, attention_model)

        result_path = os.path.join(RESULT_FOLDER, "segment_result.png")
        generate_overlay(image_resized, mask, "Segmented Tumor Region", result_path)

        patient_name = request.form.get("patient_name", "Unknown Patient")
        doctor_email = session.get("user")
        date_str = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        process_time = time.time() - start_time
        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO reports (patient_name, doctor_email, result, tumor_type, stage, confidence, image_path, slice_index, process_time, status, date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'temporary', ?)",
            (patient_name, doctor_email, "Segmentation Complete", "N/A", "N/A", 0.0, filepath, slice_index, round(process_time, 2), date_str)
        )
        report_id = cur.lastrowid
        conn.commit()
        conn.close()

        return render_template(
            "segment.html",
            result="Segmentation Complete",
            image=result_path,
            process_time=round(process_time, 2),
            scan_id=report_id
        )
    return render_template("segment.html")

@app.route("/fullscan", methods=["GET", "POST"])
def fullscan_page():
    if "user" not in session:
        return redirect("/login")
    if session.get("role") != "doctor":
        return redirect("/patient_dashboard")

    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return render_template("fullscan.html", error="Upload MRI file")
        if not file.filename.endswith('.nii'):
            return render_template("fullscan.html", error="Invalid file type. Only .nii files are allowed.")

        import tempfile
        import time
        start_time = time.time()
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            file.save(tmp.name)
            filepath = tmp.name

        # LOAD MRI
        try:
            volume, header = load_mri(filepath)
            voxel_dims = header.get_zooms()
            voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
        except:
            if os.path.exists(filepath):
                os.remove(filepath)
            return "Invalid MRI (.nii) file"

        # Remove the uploaded file to save storage
        try:
            import gc
            gc.collect()
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass

        # Select best slice
        slice_index = find_best_slice(volume)
        image = volume[:,:,slice_index]
        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image_norm = normalize(image_resized)
        input_img = image_norm[np.newaxis,...,np.newaxis]

        # Run segmentation ensemble & Generate tumor mask
        mask = segment(input_img, unet_model, resunet_model, attention_model)
        
        tumor_pixels = np.sum(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        tumor_ratio = tumor_pixels / total_pixels
        
        # Calculate true area and spherical volume estimate
        original_w, original_h = image.shape
        pixel_area_mm2 = (original_w / IMG_SIZE) * (original_h / IMG_SIZE) * voxel_dims[0] * voxel_dims[1]
        tumor_area_mm2 = tumor_pixels * pixel_area_mm2
        radius_mm = np.sqrt(tumor_area_mm2 / np.pi) if tumor_area_mm2 > 0 else 0
        tumor_volume_cm3 = (4/3 * np.pi * (radius_mm ** 3)) / 1000

        # CROP TUMOR
        tumor_crop = crop_tumor(image_resized, mask)

        # CLASSIFICATION & STAGE
        if tumor_ratio > 0.01:
            detection = "Tumor Detected"
            tumor_type, confidence = classify(tumor_crop, classifier_model, classes)
            stage = estimate_stage(tumor_volume_cm3)
            side_effects_info = get_side_effects(tumor_type, stage)
        else:
            detection = "No Tumor Detected"
            tumor_type = "None"
            confidence = 0.0
            stage = "None"
            side_effects_info = get_side_effects(None, None)
            mask = np.zeros_like(mask)

        # Generate overlay visualization
        title = f"{detection} | {tumor_type} | {stage}"
        result_path = os.path.join(RESULT_FOLDER, "fullscan_result.png")
        generate_overlay(image_resized, mask, title, result_path)

        patient_name = request.form.get("patient_name", "Unknown Patient")
        doctor_email = session.get("user")
        date_str = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        process_time = time.time() - start_time
        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        
        # Progression Alert Logic
        progression_alert = False
        if stage != "None" and stage != "N/A":
            cur.execute("SELECT stage FROM reports WHERE patient_name=? AND status='approved' ORDER BY date DESC LIMIT 1", (patient_name,))
            prev = cur.fetchone()
            if prev and prev[0] and prev[0] not in ["None", "N/A"]:
                try:
                    prev_stage_num = int(prev[0].split()[-1])
                    curr_stage_num = int(stage.split()[-1])
                    if curr_stage_num > prev_stage_num:
                        progression_alert = True
                except:
                    pass

        cur.execute(
            "INSERT INTO reports (patient_name, doctor_email, result, tumor_type, stage, confidence, image_path, slice_index, process_time, status, date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
            (patient_name, doctor_email, detection, tumor_type, stage, confidence, result_path, slice_index, round(process_time, 2), date_str)
        )
        report_id = cur.lastrowid
        conn.commit()
        conn.close()

        # Generate clinical report
        report_data_dict = {
            "patient_name": patient_name,
            "date_str": date_str,
            "scan_type": "MRI",
            "organ": "Brain",
            "detection": detection,
            "tumor_type": tumor_type,
            "confidence": round(confidence, 2),
            "mask": mask,
            "image_shape": image.shape,
            "stage": stage,
            "volume": tumor_volume_cm3
        }
        clinical_report = generate_clinical_report(report_data_dict)

        # Save data to session for the report
        session["report_data"] = {
            "clinical_report": clinical_report,
            "image": result_path
        }

        return render_template(
            "fullscan.html",
            result=detection,
            tumor_type=tumor_type,
            confidence=round(confidence,2),
            stage=stage,
            tumor_volume=round(tumor_volume_cm3, 2),
            side_effects=side_effects_info,
            clinical_report=clinical_report,
            image=result_path,
            process_time=round(process_time, 2),
            report_id=report_id,
            progression_alert=progression_alert
        )

    return render_template("fullscan.html")



@app.route("/run_full_analysis/<int:scan_id>", methods=["POST"])
def run_full_analysis(scan_id):
    print("Running full analysis for scan:", scan_id)
    if "user" not in session or session.get("role") != "doctor":
        return redirect("/login")
    
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM reports WHERE id=? AND doctor_email=?", (scan_id, session["user"]))
    report = cur.fetchone()
    
    if not report or not report[0]:
        conn.close()
        return "Report not found or missing MRI file", 404
        
    filepath = report[0] # This is the original .nii file path
    
    # Run full AI pipeline on demand
    import time
    start_time = time.time()
    try:
        volume, header = load_mri(filepath)
        voxel_dims = header.get_zooms()
        voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
        
        slice_index = find_best_slice(volume)
        image = volume[:,:,slice_index]
        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image_norm = normalize(image_resized)
        input_img = image_norm[np.newaxis,...,np.newaxis]

        mask = segment(input_img, unet_model, resunet_model, attention_model)
        
        tumor_pixels = np.sum(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        tumor_ratio = tumor_pixels / total_pixels
        
        # Calculate true area and spherical volume estimate
        original_w, original_h = image.shape
        pixel_area_mm2 = (original_w / IMG_SIZE) * (original_h / IMG_SIZE) * voxel_dims[0] * voxel_dims[1]
        tumor_area_mm2 = tumor_pixels * pixel_area_mm2
        radius_mm = np.sqrt(tumor_area_mm2 / np.pi) if tumor_area_mm2 > 0 else 0
        tumor_volume_cm3 = (4/3 * np.pi * (radius_mm ** 3)) / 1000

        tumor_crop = crop_tumor(image_resized, mask)

        if tumor_ratio > 0.01:
            detection = "Tumor Detected"
            tumor_type, confidence = classify(tumor_crop, classifier_model, classes)
            stage = estimate_stage(tumor_volume_cm3)
        else:
            detection = "No Tumor Detected"
            tumor_type = "None"
            confidence = 0.0
            stage = "None"
            mask = np.zeros_like(mask)

        title = f"{detection} | {tumor_type} | {stage}"
        result_path = os.path.join(RESULT_FOLDER, f"result_{scan_id}.png")
        generate_overlay(image_resized, mask, title, result_path)
        
        process_time = time.time() - start_time
        
        # When doctor runs full analysis on their own scan, keep it unapproved initially so it shows in pending queue
        cur.execute(
            "UPDATE reports SET status='approved', result=?, tumor_type=?, stage=?, confidence=?, image_path=?, slice_index=?, process_time=? WHERE id=?", 
            (detection, tumor_type, stage, round(confidence, 2), result_path, slice_index, round(process_time, 2), scan_id)
        )
        conn.commit()
    except Exception as e:
        print(f"Error processing report: {e}")
        conn.close()
        return "Error processing report", 500
    finally:
        try:
            conn.close()
        except:
            pass
        try:
            # The original .nii file can be removed after full analysis and result image generation
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
    
    return redirect(f"/report/{scan_id}")

@app.route("/approve_report/<int:report_id>", methods=["POST"])
def approve_report(report_id):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT id FROM reports WHERE id=?", (report_id,))
    report = cur.fetchone()

    if not report:
        conn.close()
        return "Report not found", 404

    try:
        # mark as approved and ensure it appears in patient reports
        cur.execute("""
            UPDATE reports
            SET is_approved = 1
            WHERE id = ?
        """, (report_id,))

        conn.commit()

    except Exception as e:
        print(e)
        conn.close()
        return "Error approving report", 500
        
    conn.close()
    return redirect("/doctor_dashboard")

@app.route("/reject_report/<int:report_id>", methods=["POST"])
def reject_report(report_id):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    try:
        # delete report or mark as rejected
        cur.execute("""
            DELETE FROM reports
            WHERE id = ?
        """, (report_id,))

        conn.commit()

    except Exception as e:
        print(e)
        conn.close()
        return "Error rejecting report", 500
        
    conn.close()
    return redirect("/doctor_dashboard")

@app.route("/discard_scan/<int:report_id>", methods=["POST"])
def discard_scan(report_id):
    if "user" not in session or session.get("role") != "doctor":
        return redirect("/login")
    
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM reports WHERE id=? AND doctor_email=?", (report_id, session["user"]))
    report = cur.fetchone()
    if report and report[0]:
        try:
            import os
            if os.path.exists(report[0]): # This image_path is the original .nii file
                os.remove(report[0])
            # Also remove the temporary overlay image if it exists
            temp_overlay_path = os.path.join(RESULT_FOLDER, "fullscan_result.png") # Assuming a generic name for temporary
            if os.path.exists(temp_overlay_path):
                os.remove(temp_overlay_path)
            temp_segment_overlay_path = os.path.join(RESULT_FOLDER, "segment_result.png") # Assuming a generic name for temporary
            if os.path.exists(temp_segment_overlay_path):
                os.remove(temp_segment_overlay_path)
        except Exception as e:
            print(f"Error removing file: {e}")
            pass
    cur.execute("DELETE FROM reports WHERE id=? AND doctor_email=?", (report_id, session["user"]))
    conn.commit()
    conn.close()
    
    return redirect(url_for("doctor_dashboard")) # Redirect to doctor_dashboard or analyzer_page

# =========================
# DASHBOARD
# =========================

@app.route("/doctor_dashboard")
def doctor_dashboard():
    if "user" not in session:
        return redirect("/login")
    if session.get("role") != "doctor":
        return redirect("/patient_dashboard")
        
    doctor_email = session.get("user")
    
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT id, patient_name, doctor_email, result, tumor_type, stage, confidence, image_path, slice_index, date, status, is_approved FROM reports WHERE doctor_email=? AND status!='temporary' ORDER BY date DESC", (doctor_email,))
    all_reports = cur.fetchall()
    conn.close()
    
    approved_reports = [r for r in all_reports if len(r) > 11 and r[11] == 1]
    pending_reports = [r for r in all_reports if len(r) > 11 and r[11] == 0]
    
    total_scans = len(approved_reports) 
    tumors_detected = sum(1 for r in approved_reports if r[3] == "Tumor Detected")
    high_risk_cases = sum(1 for r in approved_reports if r[5] == "Stage 4")
    
    total_confidence = sum(r[6] for r in approved_reports if r[3] == "Tumor Detected")
    avg_confidence = round(total_confidence / tumors_detected, 2) if tumors_detected > 0 else 0
    
    tumor_counts = {"glioma": 0, "meningioma": 0, "pituitary": 0}
    for r in approved_reports:
        if r[4] and r[4].lower() in tumor_counts:
            tumor_counts[r[4].lower()] += 1
            
    date_counts = {}
    for r in approved_reports:
        d = r[9][:10]
        date_counts[d] = date_counts.get(d, 0) + 1
        
    sorted_dates = sorted(date_counts.keys())
    trend_labels = sorted_dates[-7:]
    trend_data = [date_counts[d] for d in trend_labels]

    critical_alerts = [r for r in approved_reports if r[5] == "Stage 4"]
    recent_reports = approved_reports[:10]
    
    return render_template(
        "doctor_dashboard.html",
        recent_reports=recent_reports,
        pending_reports=pending_reports,
        total_scans=total_scans,
        tumors_detected=tumors_detected,
        high_risk_cases=high_risk_cases,
        avg_confidence=avg_confidence,
        tumor_labels=list(tumor_counts.keys()),
        tumor_data=list(tumor_counts.values()),
        trend_labels=trend_labels,
        trend_data=trend_data,
        critical_alerts=critical_alerts
    )

@app.route("/delete_report/<int:report_id>", methods=["POST"])
def delete_report(report_id):
    if "user" not in session or session.get("role") != "doctor":
        return redirect("/login")
        
    doctor_email = session.get("user")
    
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT image_path FROM reports WHERE id=? AND doctor_email=?", (report_id, doctor_email))
    report = cur.fetchone()
    
    if report:
        image_path = report[0]
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass
        
        cur.execute("DELETE FROM reports WHERE id=? AND doctor_email=?", (report_id, doctor_email))
        conn.commit()
    conn.close()
    
    return redirect(request.referrer or url_for('doctor_dashboard'))

@app.route("/export_csv")
def export_csv():
    if "user" not in session or session.get("role") != "doctor":
        return redirect("/login")
        
    doctor_email = session.get("user")
    patient_name = request.args.get("patient")
    
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    
    if patient_name:
        cur.execute("SELECT id, patient_name, doctor_email, result, tumor_type, stage, confidence, date FROM reports WHERE doctor_email=? AND patient_name=? AND is_approved=1 ORDER BY date DESC", (doctor_email, patient_name))
    else:
        cur.execute("SELECT id, patient_name, doctor_email, result, tumor_type, stage, confidence, date FROM reports WHERE doctor_email=? AND is_approved=1 ORDER BY date DESC", (doctor_email,))
        
    reports = cur.fetchall()
    conn.close()
    
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(["ID", "Patient Name", "Doctor Email", "Result", "Tumor Type", "Stage", "Confidence %", "Date"])
    cw.writerows(reports)
    
    output = si.getvalue()
    filename = f"reports_export_{patient_name}.csv" if patient_name else "reports_export_all.csv"
    
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )

@app.route("/analyzer")
def analyzer():
    if "user" not in session:
        return redirect("/login")
    if session.get("role") != "doctor":
        return redirect("/patient_dashboard")
    return render_template("analyzer.html")

@app.route("/patients")
def patients():
    if "user" not in session or session.get("role") != "doctor":
        flash("Unauthorized access.", "danger")
        return redirect(url_for("login"))
    
    doctor_email = session["user"]
    
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT id, patient_name, doctor_email, result, tumor_type, stage, confidence, image_path, slice_index, date FROM reports WHERE doctor_email=? AND is_approved=1 ORDER BY date DESC", (doctor_email,))
    reports = cur.fetchall()
    conn.close()
    
    # Group reports by patient name
    grouped_patients = {}
    for r in reports:
        p_name = r[1]
        if p_name not in grouped_patients:
            grouped_patients[p_name] = []
        grouped_patients[p_name].append(r)
    
    return render_template("patients.html", grouped_patients=grouped_patients)

@app.route("/patient/<name>")
def patient_detail(name):
    if "user" not in session or session.get("role") != "doctor":
        flash("Unauthorized access.", "danger")
        return redirect(url_for("login"))
        
    doctor_email = session["user"]
    
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    # Order ascending for logical timeline progression (oldest -> newest)
    cur.execute("SELECT id, patient_name, doctor_email, result, tumor_type, stage, confidence, image_path, slice_index, date FROM reports WHERE patient_name=? AND doctor_email=? AND is_approved=1 ORDER BY date ASC", (name, doctor_email))
    patient_reports = cur.fetchall()
    conn.close()
    
    if not patient_reports:
        flash("Patient not found or unauthorized.", "danger")
        return redirect(url_for("patients"))
        
    return render_template("patient_detail.html", patient_name=name, reports=patient_reports)

@app.route("/patient_dashboard", methods=["GET", "POST"])
def patient_dashboard():
    if "user" not in session:
        return redirect("/login")
    if session.get("role") == "doctor":
        return redirect("/doctor_dashboard")
        
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return "Upload MRI file"

        import tempfile
        import time
        start_time = time.time()
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            file.save(tmp.name)
            filepath = tmp.name

        try:
            volume, header = load_mri(filepath)
            voxel_dims = header.get_zooms()
            voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
        except:
            if os.path.exists(filepath):
                os.remove(filepath)
            flash("Invalid MRI (.nii) file", "danger")
            return redirect(url_for("patient_dashboard"))

        try:
            import gc
            gc.collect()
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass

        slice_index = find_best_slice(volume)
        image = volume[:,:,slice_index]
        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image_norm = normalize(image_resized)
        input_img = image_norm[np.newaxis,...,np.newaxis]

        mask = segment(input_img, unet_model, resunet_model, attention_model)
        
        tumor_pixels = np.sum(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        tumor_ratio = tumor_pixels / total_pixels
        
        # Calculate true area and spherical volume estimate
        original_w, original_h = image.shape
        pixel_area_mm2 = (original_w / IMG_SIZE) * (original_h / IMG_SIZE) * voxel_dims[0] * voxel_dims[1]
        tumor_area_mm2 = tumor_pixels * pixel_area_mm2
        radius_mm = np.sqrt(tumor_area_mm2 / np.pi) if tumor_area_mm2 > 0 else 0
        tumor_volume_cm3 = (4/3 * np.pi * (radius_mm ** 3)) / 1000

        tumor_crop = crop_tumor(image_resized, mask)

        if tumor_ratio > 0.01:
            detection = "Tumor Detected"
            tumor_type, confidence = classify(tumor_crop, classifier_model, classes)
            stage = estimate_stage(tumor_volume_cm3)
        else:
            detection = "No Tumor Detected"
            tumor_type = "None"
            confidence = 0.0
            stage = "None"

        title = f"{detection} | {tumor_type} | {stage}"
        patient_email = session.get("user")
        
        import sqlite3
        from datetime import datetime
        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        
        cur.execute("SELECT name FROM users WHERE email=?", (patient_email,))
        user_row = cur.fetchone()
        patient_name = user_row[0] if user_row else patient_email
            
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cur.execute("SELECT email FROM users WHERE role='doctor' LIMIT 1")
        doc_row = cur.fetchone()
        
        # Override doctor_email to specifically isolate this from doctor dashboards
        doctor_email = "patient_self"
        
        cur.execute(
            "INSERT INTO reports (patient_name, doctor_email, result, tumor_type, stage, confidence, image_path, slice_index, process_time, status, date, is_approved) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'patient_upload', ?, 1)",
            (patient_name, doctor_email, detection, tumor_type, stage, round(confidence, 2), "", slice_index, round(time.time() - start_time, 2), date_str)
        )
        report_id = cur.lastrowid
        
        result_path = os.path.join(RESULT_FOLDER, f"result_{report_id}.png")
        generate_overlay(image_resized, mask, title, result_path)
        
        cur.execute("UPDATE reports SET image_path=? WHERE id=?", (result_path, report_id))
        conn.commit()
        conn.close()

        flash("Your scan has been successfully processed and final results are saved.", "success")
        return redirect(url_for("patient_dashboard"))

    # GET REQUEST HANDLING
    patient_email = session.get("user")
    import sqlite3
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT name FROM users WHERE email=?", (patient_email,))
    user_row = cur.fetchone()
    patient_name = user_row[0] if user_row else patient_email

    # Provide only reports where is_approved=1 for the patient dashboard timeline
    cur.execute("SELECT id, result, tumor_type, stage, confidence, image_path, date FROM reports WHERE patient_name=? AND is_approved=1 ORDER BY date DESC", (patient_name,))
    reports = cur.fetchall()
    
    conn.close()

    latest_report = reports[0] if reports else None
    
    chart_labels = []
    chart_data = []
    for r in reversed(reports):
        chart_labels.append(r[6][:10])
        stage_str = r[3]
        if stage_str == "None" or stage_str == "N/A":
            val = 0
        else:
            try:
                val = int(stage_str.split()[-1])
            except:
                val = 1
        chart_data.append(val)
        
    # Generate simple explanation for latest report
    explanation = None
    symptoms = None
    if latest_report:
        tumor_type = latest_report[2]
        stage = latest_report[3]
        if "Tumor Detected" in latest_report[1]:
            explanation = f"An anomaly consistent with {tumor_type.title()} was detected."
            symptoms = get_side_effects(tumor_type, stage)
        else:
            explanation = "The scan appears normal based on AI analysis."
            symptoms = "None"

    return render_template(
        "patient_dashboard.html",
        patient_name=patient_name,
        reports=reports,
        latest_report=latest_report,
        chart_labels=chart_labels,
        chart_data=chart_data,
        explanation=explanation,
        symptoms=symptoms
    )

@app.route("/report/<int:report_id>")
def view_report(report_id):
    if "user" not in session:
        return redirect("/login")
    if session.get("role") != "doctor":
        return redirect("/patient_dashboard")
        
    doctor_email = session.get("user")
    
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT id, patient_name, doctor_email, result, tumor_type, stage, confidence, image_path, slice_index, date, is_approved FROM reports WHERE id=? AND doctor_email=?", (report_id, doctor_email))
    report = cur.fetchone()
    conn.close()
    
    if not report:
        return "Report not found or access denied.", 404
        
    side_effects_info = get_side_effects(report[4], report[5]) if report[4] not in ['N/A', 'None'] else "No clinical notes applicable."
    
    view_data = {
        "id": report[0],
        "patient_name": report[1],
        "doctor_email": report[2],
        "result": report[3],
        "tumor_type": report[4],
        "stage": report[5],
        "confidence": report[6],
        "image_path": report[7],
        "slice_index": report[8],
        "date": report[9],
        "is_approved": report[10],
        "clinical_notes": side_effects_info
    }
    
    return render_template("report.html", report=view_data)

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")
    if session.get("role") == "doctor":
        return redirect("/doctor_dashboard")
    return redirect("/patient_dashboard")


# =========================
# HOME
# =========================

@app.route("/")
def home():
    return redirect("/login")


# =========================
# PDF REPORT GENERATOR
# =========================

@app.route("/generate_report")
def generate_report():
    data = session.get("report_data")
    if not data or "clinical_report" not in data:
        return "No report data found", 404
        
    clinical_report = data["clinical_report"]
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 750, "AI Clinical Diagnostic Report")
    
    # Patient Info & Detection
    c.setFont("Helvetica", 12)
    p_info = clinical_report.get("patient_info", {})
    c.drawString(50, 710, f"Patient Name: {p_info.get('name')} | Date: {p_info.get('scan_date')}")
    c.drawString(50, 690, f"Scan Type: {p_info.get('scan_type')} | Organ: {p_info.get('organ')}")
    
    det_sum = clinical_report.get("detection_summary", {})
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 660, f"Detection Summary: {det_sum.get('tumor_detected')}")
    c.setFont("Helvetica", 12)
    c.drawString(50, 640, f"Tumor Type: {det_sum.get('tumor_type')}  |  Confidence: {det_sum.get('confidence')}")

    y = 610
    
    if det_sum.get("tumor_detected") == "Detected":
        chars = clinical_report.get("tumor_characteristics", {})
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Tumor Characteristics")
        c.setFont("Helvetica", 12)
        y -= 20
        c.drawString(50, y, f"Location: {chars.get('location')} ({chars.get('region')})")
        y -= 20
        c.drawString(50, y, f"Estimated Volume: {chars.get('volume_cm3')} cm³")
        y -= 20
        c.drawString(50, y, f"Bounding Box: {chars.get('bbox')}")
        y -= 30
        
        sev = clinical_report.get("severity_assessment", {})
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Severity Stage: {sev.get('stage')} (Risk: {sev.get('risk_level')})")
        c.setFont("Helvetica", 12)
        y -= 30
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Clinical Findings:")
    c.setFont("Helvetica", 12)
    y -= 20
    from reportlab.lib.utils import simpleSplit
    lines = simpleSplit(clinical_report.get("clinical_findings", ""), "Helvetica", 12, 500)
    for line in lines:
        c.drawString(50, y, line)
        y -= 20
        if y < 100: c.showPage(); y = 750
        
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Symptoms & Clinical Correlation:")
    c.setFont("Helvetica", 12)
    y -= 20
    c.drawString(50, y, clinical_report.get("symptoms", "None"))
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Recommendations:")
    c.setFont("Helvetica", 12)
    y -= 20
    for rec in clinical_report.get("recommendations", []):
        c.drawString(60, y, f"- {rec}")
        y -= 20
        if y < 100: c.showPage(); y = 750
        
    # Include the annotated image (For Doctors)
    is_doctor = session.get("role") == "doctor"
    if is_doctor:
        image_path = data.get("image")
        if image_path and os.path.exists(image_path):
            if y < 250:
                c.showPage()
                y = 800
            c.drawImage(image_path, 50, y - 220, width=200, height=200)
        
    c.save()
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name="MRI_Clinical_Report.pdf", mimetype="application/pdf")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True)


# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     app.run(host="0.0.0.0", port=port)

