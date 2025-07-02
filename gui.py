import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox, filedialog
from datetime import datetime
import csv
import os

log_file = "intrusion_log.csv"

# -------------------- GUI Dashboard --------------------

def log_intrusion(vehicle_id, timestamp, attack_type):
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Vehicle ID', 'Timestamp', 'Attack Type'])
        writer.writerow([vehicle_id, timestamp, attack_type])

def predict_intrusion():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        df_input = pd.read_csv(file_path)
        print("Loaded CSV shape:", df_input.shape)

        if 'label' in df_input.columns:
            df_input = df_input.drop('label', axis=1)

        for col in ['protocol_type', 'service', 'flag']:
            if col in df_input.columns:
                df_input[col] = LabelEncoder().fit_transform(df_input[col])

        print("Processed columns:", df_input.columns)

        X_input = df_input.values
        X_input_scaled = MinMaxScaler().fit_transform(X_input)
        X_input_dl = X_input_scaled.reshape(X_input_scaled.shape[0], X_input_scaled.shape[1], 1)

        model = load_model("vanet_cnn_lstm_model.h5")
        predictions = (model.predict(X_input_dl) > 0.5).astype("int32")
        print("Predictions:", predictions[:10])

        normal = np.sum(predictions == 0)
        attack = np.sum(predictions == 1)

        # Log and display live alerts
        alerts_box.delete('1.0', tk.END)
        for idx, pred in enumerate(predictions):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            vehicle_id = f"V{1000+idx}"
            if pred[0] == 1:
                alerts_box.insert(tk.END, f" {timestamp} | {vehicle_id} | Attack Detected\n")
                log_intrusion(vehicle_id, timestamp, "Attack")
            else:
                alerts_box.insert(tk.END, f" {timestamp} | {vehicle_id} | Normal\n")
                log_intrusion(vehicle_id, timestamp, "Normal")

        result_label.config(text=f" Normal: {normal} |  Attack: {attack}")
        messagebox.showinfo("Prediction Complete", f"Normal: {normal}\nAttack: {attack}")

        # -------------------- Plotting --------------------
        plt.figure(figsize=(5, 5))
        labels = ['Normal', 'Attack']
        sizes = [normal, attack]
        colors = ['#00cc99', '#ff6666']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title("Intrusion Detection Results")
        plt.axis('equal')
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))
        print("Error occurred:", e)

# -------------------- Tkinter GUI --------------------

root = tk.Tk()
root.title("Real-Time VANET Intrusion Detection Dashboard")
root.geometry("700x600")
root.config(bg="#f0f8ff")

title = tk.Label(root, text="Real-Time VANET Intrusion Detection", font=("Helvetica", 18, "bold"), bg="#f0f8ff", fg="#333")
title.pack(pady=10)

upload_btn = tk.Button(root, text="Upload CSV & Predict", font=("Helvetica", 14), bg="#007acc", fg="white", padx=20, pady=10, command=predict_intrusion)
upload_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#f0f8ff")
result_label.pack(pady=10)

alerts_label = tk.Label(root, text="Live Intrusion Alerts:", font=("Helvetica", 14, "bold"), bg="#f0f8ff")
alerts_label.pack()

alerts_box = tk.Text(root, height=15, width=80, font=("Consolas", 10))
alerts_box.pack(pady=10)

exit_btn = tk.Button(root, text="Exit", font=("Helvetica", 12), bg="#cc0000", fg="white", command=root.destroy)
exit_btn.pack(side="bottom", pady=10)

root.mainloop()
