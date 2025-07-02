'''Feature Name	Description
1. Vehicle Speed	Current speed of the vehicle (km/h)
2. Acceleration	Acceleration or deceleration rate
3. GPS Coordinates	Latitude or longitude (can be split as two features)
4. Signal Strength	Signal power level (dB) of received messages
5. Packet Sending Rate	How many packets are sent per unit time
6. Message Type	Type of VANET message (e.g., beacon, alert)
7. Timestamp Difference	Time difference between consecutive packets
8. Distance to Neighboring Vehicles	Estimated distance to closest vehicles
9. Packet Size	Size of message packet (bytes)
10. Number of Retransmissions	How many times a message is retransmitted'''
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import datetime
import pandas as pd
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

MODEL_PATH = "vanet_cnn_model.h5"


# Dummy training function to create and save the CNN model
def train_and_save_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists, skipping training.")
        return

    print("Training model...")
    # Dummy data: 1000 samples, 10 features each (like VANET packet features)
    X_train = np.random.rand(1000, 10, 1).astype(np.float32)
    y_train = np.random.randint(0, 2, 1000)  # binary labels: 0 = safe, 1 = intrusion

    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(10, 1)),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=2)

    model.save(MODEL_PATH)
    print("Model trained and saved.")


# Load model (train if not exists)
def load_vanet_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    return load_model(MODEL_PATH)


# Predict intrusion or safe from input features
def predict_intrusion(model, features):
    # features: list or array of length 10
    x = np.array(features).reshape(1, 10, 1).astype(np.float32)
    pred = model.predict(x)[0][0]
    return "Intrusion Detected" if pred >= 0.5 else "Safe"


# Blockchain log stub
def log_to_blockchain(vehicle_id, timestamp, result):
    txid = "tx_" + ''.join(random.choices('0123456789abcdef', k=12))
    print(f"[Blockchain] Vehicle:{vehicle_id}, Time:{timestamp}, Result:{result}, TxID:{txid}")
    return txid


class VANETApp:
    def __init__(self, root):
        self.root = root
        self.root.title(" VANET Intrusion Detection (CNN Model)")
        self.root.state('zoomed')  # Fullscreen
        self.root.configure(bg="#2c3e50")

        # Load background image (optional)
        try:
            bg_img = Image.open("background.jpg").resize(
                (self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
            self.bg_photo = ImageTk.PhotoImage(bg_img)
            bg_label = tk.Label(self.root, image=self.bg_photo)
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except:
            pass  # no bg image available

        self.model = load_vanet_model()
        self.logs = []

        # Frame for inputs and buttons
        self.frame = tk.Frame(root, bg="#34495e", bd=5)
        self.frame.place(relx=0.5, rely=0.5, anchor='center')

        label_style = {"font": ("Helvetica", 14), "bg": "#34495e", "fg": "white"}
        entry_style = {"font": ("Helvetica", 12)}

        # Input fields for 10 features (simulate VANET packet)
        self.entries = []
        for i in range(10):
            lbl = tk.Label(self.frame, text=f"Feature {i + 1}:", **label_style)
            lbl.grid(row=i, column=0, sticky='e', padx=5, pady=5)
            entry = tk.Entry(self.frame, **entry_style, width=10)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries.append(entry)

        # Vehicle ID input
        lbl_vid = tk.Label(self.frame, text="Vehicle ID:", **label_style)
        lbl_vid.grid(row=10, column=0, sticky='e', padx=5, pady=10)
        self.vehicle_entry = tk.Entry(self.frame, **entry_style, width=15)
        self.vehicle_entry.grid(row=10, column=1, padx=5, pady=10)

        # Buttons
        btn_style = {"font": ("Helvetica", 13, "bold"), "width": 15, "bd": 0, "fg": "white"}

        self.detect_btn = tk.Button(self.frame, text=" Detect Intrusion", bg="#27ae60", activebackground="#2ecc71",
                                    command=self.detect, **btn_style)
        self.detect_btn.grid(row=11, column=0, columnspan=2, pady=(10, 5))

        self.log_btn = tk.Button(self.frame, text="Log to Blockchain", bg="#2980b9", activebackground="#3498db",
                                 command=self.log_intrusion, state=tk.DISABLED, **btn_style)
        self.log_btn.grid(row=12, column=0, columnspan=2, pady=5)

        self.export_btn = tk.Button(self.frame, text=" Export Logs", bg="#f39c12", activebackground="#f1c40f",
                                    command=self.export_logs, **btn_style)
        self.export_btn.grid(row=13, column=0, columnspan=2, pady=5)

        # Result label
        self.result_label = tk.Label(self.frame, text="", font=("Helvetica", 16, "bold"), bg="#34495e", fg="white")
        self.result_label.grid(row=14, column=0, columnspan=2, pady=10)

    def detect(self):
        try:
            features = [float(e.get()) for e in self.entries]
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all features.")
            return

        vehicle_id = self.vehicle_entry.get().strip()
        if not vehicle_id:
            messagebox.showerror("Input Error", "Please enter a Vehicle ID.")
            return

        result = predict_intrusion(self.model, features)
        self.result_label.config(text=f"Result: {result}", fg="#e74c3c" if "Intrusion" in result else "#2ecc71")
        self.last_detection = (vehicle_id, result, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.log_btn.config(state=tk.NORMAL if "Intrusion" in result else tk.DISABLED)

    def log_intrusion(self):
        vehicle_id, result, timestamp = self.last_detection
        txid = log_to_blockchain(vehicle_id, timestamp, result)

        self.logs.append({
            "timestamp": timestamp,
            "vehicle_id": vehicle_id,
            "result": result,
            "txid": txid
        })

        messagebox.showinfo("Blockchain Log", f"Intrusion logged on blockchain.\nTxID: {txid}")
        self.log_btn.config(state=tk.DISABLED)

    def export_logs(self):
        if not self.logs:
            messagebox.showwarning("No Logs", "No logs to export.")
            return
        df = pd.DataFrame(self.logs)
        df.to_csv("vanet_intrusion_logs.csv", index=False)
        messagebox.showinfo("Export Success", "Logs exported to vanet_intrusion_logs.csv")


if __name__ == "__main__":
    root = tk.Tk()
    app = VANETApp(root)
    root.mainloop()
