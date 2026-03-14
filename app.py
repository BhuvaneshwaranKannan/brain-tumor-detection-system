import os
import sqlite3
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, session
from tensorflow.keras.models import load_model



# =========================
# APP CONFIG
# =========================

app = Flask(__name__)
app.secret_key = "brain_tumor_project"

IMG_SIZE = 128

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static"
MODEL_FOLDER = "model"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
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
        password TEXT
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
# LOAD SEGMENTATION MODELS
# =========================

def load_safe_model(path):

    if os.path.exists(path):
        print("Loading:", path)
        return load_model(path, custom_objects=custom)

    print("Missing model:", path)
    return None


unet_model = load_safe_model(os.path.join(MODEL_FOLDER, "x1.h5"))
resunet_model = load_safe_model(os.path.join(MODEL_FOLDER, "y1.h5"))
attention_model = load_safe_model(os.path.join(MODEL_FOLDER, "z1.h5"))


# =========================
# LOAD CLASSIFIER
# =========================

classifier_model = load_model(os.path.join(MODEL_FOLDER, "tumor_type_model.h5"))

classes = ["glioma", "meningioma", "pituitary"]


# =========================
# HELPER FUNCTIONS
# =========================

def normalize(img):

    mean = np.mean(img)
    std = np.std(img)

    if std != 0:
        img = (img - mean) / std

    return img


def postprocess(mask):

    mask = mask.astype(np.uint8)

    # remove noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # keep largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels > 1:

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)

    # fill holes
    mask_flood = mask.copy()

    h, w = mask.shape
    flood_mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(mask_flood, flood_mask, (0,0), 1)

    mask_inv = cv2.bitwise_not(mask_flood)

    mask = mask | mask_inv

    return mask


# =========================
# STAGE CLASSIFICATION
# =========================

def classify_stage(tumor_pixels, img_size):

    total_pixels = img_size * img_size

    tumor_ratio = tumor_pixels / total_pixels

    if tumor_ratio < 0.02:
        return "Stage 1"
    elif tumor_ratio < 0.05:
        return "Stage 2"
    elif tumor_ratio < 0.10:
        return "Stage 3"
    else:
        return "Stage 4"


# =========================
# FIND BEST SLICE
# =========================

def find_best_slice(volume):

    max_var = 0
    best_index = volume.shape[2] // 2

    for i in range(volume.shape[2]):

        slice_img = volume[:,:,i]
        var = np.var(slice_img)

        if var > max_var:
            max_var = var
            best_index = i

    return best_index


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
            "SELECT * FROM users WHERE email=? AND password=?",
            (email,password)
        )

        user = cur.fetchone()
        conn.close()

        if user:
            session["user"] = email
            return redirect("/dashboard")

        return "Invalid Login"

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

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()

        try:

            cur.execute(
                "INSERT INTO users(name,email,password) VALUES(?,?,?)",
                (name,email,password)
            )

            conn.commit()

        except:
            return "User already exists"

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
# DASHBOARD
# =========================

@app.route("/dashboard", methods=["GET","POST"])
def dashboard():

    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":

        file = request.files["file"]

        if file.filename == "":
            return "Upload MRI file"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)


        # =========================
        # LOAD MRI
        # =========================

        try:
            volume = nib.load(filepath).get_fdata()
        except:
            return "Invalid MRI (.nii) file"


        slice_index = find_best_slice(volume)

        image = volume[:,:,slice_index]

        image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))

        image = normalize(image)

        input_img = image[np.newaxis,...,np.newaxis]


        # =========================
        # MULTI-SLICE CLASSIFICATION
        # =========================

        slice_ids = [slice_index-1, slice_index, slice_index+1]

        predictions = []

        for sid in slice_ids:

            sid = max(0, min(sid, volume.shape[2]-1))

            slice_img = volume[:,:,sid]

            slice_img = cv2.resize(slice_img,(128,128))

            slice_img = cv2.normalize(slice_img,None,0,255,cv2.NORM_MINMAX)

            slice_img = cv2.cvtColor(slice_img.astype("float32"),cv2.COLOR_GRAY2RGB)

            slice_img = slice_img / 255.0

            slice_img = np.expand_dims(slice_img,axis=0)

            pred = classifier_model.predict(slice_img)

            predictions.append(pred[0])


        avg_pred = np.mean(predictions, axis=0)

        class_index = np.argmax(avg_pred)

        tumor_type = classes[class_index]

        confidence = float(np.max(avg_pred))*100


        # =========================
        # SEGMENTATION
        # =========================

        preds = []

        if unet_model:
            preds.append(unet_model.predict(input_img)[0,:,:,0])

        if resunet_model:
            preds.append(resunet_model.predict(input_img)[0,:,:,0])

        if attention_model:
            preds.append(attention_model.predict(input_img)[0,:,:,0])


        final_pred = np.mean(preds, axis=0)

        final_pred = cv2.GaussianBlur(final_pred,(5,5),0)

        mask = (final_pred > 0.45).astype(np.uint8)

        mask = postprocess(mask)


        tumor_pixels = np.sum(mask)


        if tumor_pixels > 50:

            detection = "Tumor Detected"

            stage = classify_stage(tumor_pixels, IMG_SIZE)

        else:

            detection = "No Tumor Detected"

            stage = "None"


        # =========================
        # VISUALIZATION
        # =========================

        plt.figure(figsize=(6,6))

        plt.imshow(image, cmap="gray")

        plt.imshow(mask, cmap="jet", alpha=0.4)

        plt.title(f"{detection} | {tumor_type} | {stage}")

        plt.axis("off")

        result_path = os.path.join(RESULT_FOLDER,"result.png")

        plt.savefig(result_path, bbox_inches="tight", pad_inches=0)

        plt.close()


        return render_template(
            "dashboard.html",
            result=detection,
            tumor_type=tumor_type,
            confidence=round(confidence,2),
            stage=stage,
            image=result_path
        )


    return render_template("dashboard.html")


# =========================
# HOME
# =========================

@app.route("/")
def home():
    return redirect("/login")


# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True)
