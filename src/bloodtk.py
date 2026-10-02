import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# ---------------- GLOBALS ----------------
train_path = ""
test_path = ""
model = None
history = None
class_map = {}

ANNOTATION_PATH = filedialog.askdirectory(title="Select Annotations Folder")

# ---------------- WINDOW ----------------
root = tk.Tk()
root.title("Blood Cell Analyzer")
root.geometry("1200x700")
root.configure(bg="#1e1e2f")

# ---------------- HEADER ----------------
header = tk.Label(root, text="Blood Cell Classification System",
                  font=("Arial", 20, "bold"), bg="#1e1e2f", fg="white")
header.pack(pady=10)

# ---------------- LEFT PANEL ----------------
left_frame = tk.Frame(root, bg="#2c2c3e", width=300)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

# ---------------- RIGHT PANEL ----------------
right_frame = tk.Frame(root, bg="#1e1e2f")
right_frame.pack(side="right", expand=True, fill="both")

# ---------------- IMAGE PANEL ----------------
panel = tk.Label(right_frame, bg="#1e1e2f")
panel.pack(pady=20)

# ---------------- LOG BOX ----------------
log_box = ScrolledText(right_frame, width=70, height=12, bg="#121212", fg="white")
log_box.pack(pady=10)

def log(msg):
    log_box.insert(tk.END, msg + "\n")
    log_box.see(tk.END)

# ---------------- DATA ----------------
def get_data(folder):
    X, y = [], []
    class_map_local = {}

    for idx, folder_name in enumerate(os.listdir(folder)):
        path = os.path.join(folder, folder_name)
        if not os.path.isdir(path): continue

        class_map_local[idx] = folder_name

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, (80,60))
                X.append(img)
                y.append(idx)

    X = np.array(X) / 255.0
    y = to_categorical(y, num_classes=len(class_map_local))

    return X, y, class_map_local

# ---------------- UPLOAD ----------------
def upload_train():
    global train_path, test_path

    path = filedialog.askdirectory(title="Select TRAIN Folder")
    if not path:
        log("❌ No folder selected")
        return

    path = os.path.normpath(path)

    if not path.endswith("TRAIN"):
        log("⚠️ Please select TRAIN folder")
        return

    train_path = path
    log(f"✅ Train Loaded")

    parent = os.path.dirname(train_path)
    possible_test = os.path.join(parent, "TEST")

    if os.path.exists(possible_test):
        test_path = possible_test
        log(f"✅ Test Auto Loaded")
    else:
        log("❌ TEST folder not found")

# ---------------- PREPROCESS ----------------
def preprocess():
    global X_train, y_train, X_test, y_test, class_map

    if not train_path or not test_path:
        log("❌ Load TRAIN first")
        return

    log("⏳ Preprocessing...")

    X_train, y_train, class_map = get_data(train_path)
    X_test, y_test, _ = get_data(test_path)

    log(f"✅ Train images: {len(X_train)}")
    log(f"✅ Test images: {len(X_test)}")

# ---------------- MODEL ----------------
def build_model():
    model = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=(60,80,3)),
        MaxPooling2D(),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128,activation='relu'),
        Dense(len(class_map),activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    global model, history

    log("🚀 Training Started...")

    model = build_model()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=12,
        batch_size=32
    )

    model.save("model.h5")
    log("✅ Training Completed")

    # 🔥 IMPORTANT: call graph in main thread
    root.after(100, plot_graphs)

def start_training():
    threading.Thread(target=train_model).start()

# ---------------- GRAPHS ----------------
def plot_graphs():
    try:
        plt.figure()
        plt.plot(history.history['accuracy'],'r')
        plt.plot(history.history['val_accuracy'],'b')
        plt.title("Accuracy")
        plt.savefig("accuracy.png")
        plt.close()

        plt.figure()
        plt.plot(history.history['loss'],'r')
        plt.plot(history.history['val_loss'],'b')
        plt.title("Loss")
        plt.savefig("loss.png")
        plt.close()

        log("📊 Graphs Generated Successfully!")

    except Exception as e:
        log(f"❌ Graph Error: {str(e)}")

def show_graph(path):
    img = Image.open(path).resize((500,350))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

# ---------------- PREDICTION ----------------
def predict_image():
    global model

    file = filedialog.askopenfilename()
    img = cv2.imread(file)

    img = cv2.resize(img,(80,60)) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    idx = np.argmax(pred)

    log(f"🔍 Prediction: {class_map[idx]} ({np.max(pred)*100:.2f}%)")

# ---------------- DETECTION ----------------
def detect_cells():
    global ANNOTATION_PATH

    img_path = filedialog.askopenfilename(title="Select Image")

    if not ANNOTATION_PATH:
        ANNOTATION_PATH = filedialog.askdirectory(title="Select Annotations Folder")

    filename = os.path.basename(img_path).replace(".jpg",".xml")
    xml_path = os.path.join(ANNOTATION_PATH, filename)

    if not os.path.exists(xml_path):
        log(f"❌ XML not found for {filename}")
        return

    img = cv2.imread(img_path)
    tree = ET.parse(xml_path)

    for obj in tree.findall('object'):
        name = obj.find('name').text
        box = obj.find('bndbox')

        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)

        color = (0,255,0) if name.startswith("R") else (0,0,255)

        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,2)
        cv2.putText(img,name,(xmin,ymin-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize((500,350))
    img_tk = ImageTk.PhotoImage(img)

    panel.config(image=img_tk)
    panel.image = img_tk

# ---------------- LOAD MODEL ----------------
def load_model_btn():
    global model
    model = load_model("model.h5")
    log("✅ Model Loaded")

# ---------------- BUTTON STYLE ----------------
def btn(parent, text, cmd):
    return tk.Button(parent, text=text, command=cmd,
                     bg="#4CAF50", fg="white",
                     font=("Arial", 10, "bold"),
                     width=20, height=2)

# ---------------- BUTTONS ----------------
btn(left_frame, "Upload TRAIN", upload_train).pack(pady=5)
btn(left_frame, "Preprocess", preprocess).pack(pady=5)
btn(left_frame, "Train Model", start_training).pack(pady=5)
btn(left_frame, "Load Model", load_model_btn).pack(pady=5)
btn(left_frame, "Predict Image", predict_image).pack(pady=5)
btn(left_frame, "Detect Cells", detect_cells).pack(pady=5)
btn(left_frame, "Show Accuracy", lambda: show_graph("accuracy.png")).pack(pady=5)
btn(left_frame, "Show Loss", lambda: show_graph("loss.png")).pack(pady=5)

# ---------------- RUN ----------------
root.mainloop()