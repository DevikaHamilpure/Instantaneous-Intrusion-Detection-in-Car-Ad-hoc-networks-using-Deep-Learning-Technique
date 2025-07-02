import tkinter as tk
from tkinter import messagebox
import datetime
import pandas as pd
import random
from PIL import Image, ImageTk

# Intrusion detection model stub (replace with CNN-LSTM later)
def detect_intrusion(packet):
    return random.choice(["✅ Safe", "⚠️ Intrusion Detected"])

# Blockchain logging stub (replace with web3.py call)
def log_to_blockchain(vehicle_id, timestamp, result):
    print(f"[Blockchain Log] Vehicle: {vehicle_id}, Time: {timestamp}, Result: {result}")
    return "tx_hash_" + ''.join(random.choices('0123456789abcdef', k=10))

class VANETGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(" VANET Intrusion Detection Dashboard")
        self.root.state('zoomed')  # Fullscreen window

        # Set background image
        self.bg_image = Image.open("back.png")  # Ensure this image exists
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        self.bg_label = tk.Label(root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.logs = []

        # Frame to center the UI
        self.main_frame = tk.Frame(root, bg='#f0f4f7', bd=2, relief='ridge')
        self.main_frame.place(relx=0.5, rely=0.5, anchor='center')

        # Styling
        label_font = ("Arial", 12, "bold")
        entry_bg = "#ffffff"
        btn_font = ("Arial", 11, "bold")

        tk.Label(self.main_frame, text="Vehicle ID:", font=label_font, bg='#f0f4f7').grid(row=0, column=0, padx=10, pady=5, sticky='e')
        self.vehicle_entry = tk.Entry(self.main_frame, bg=entry_bg)
        self.vehicle_entry.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.main_frame, text="Speed (km/h):", font=label_font, bg='#f0f4f7').grid(row=1, column=0, padx=10, pady=5, sticky='e')
        self.speed_entry = tk.Entry(self.main_frame, bg=entry_bg)
        self.speed_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.main_frame, text="Signal Strength (dB):", font=label_font, bg='#f0f4f7').grid(row=2, column=0, padx=10, pady=5, sticky='e')
        self.signal_entry = tk.Entry(self.main_frame, bg=entry_bg)
        self.signal_entry.grid(row=2, column=1, padx=10, pady=5)

        tk.Label(self.main_frame, text="Message Type:", font=label_font, bg='#f0f4f7').grid(row=3, column=0, padx=10, pady=5, sticky='e')
        self.message_entry = tk.Entry(self.main_frame, bg=entry_bg)
        self.message_entry.grid(row=3, column=1, padx=10, pady=5)

        self.detect_button = tk.Button(self.main_frame, text=" Detect Intrusion", font=btn_font, bg="#28a745", fg="white", command=self.detect)
        self.detect_button.grid(row=4, column=0, columnspan=2, pady=10, ipadx=10)

        self.result_label = tk.Label(self.main_frame, text="", font=("Arial", 14, "bold"), bg='#f0f4f7')
        self.result_label.grid(row=5, column=0, columnspan=2, pady=5)

        self.log_button = tk.Button(self.main_frame, text="Log to Blockchain", font=btn_font, bg="#007bff", fg="white", command=self.log_intrusion, state=tk.DISABLED)
        self.log_button.grid(row=6, column=0, columnspan=2, pady=5, ipadx=10)

        self.export_button = tk.Button(self.main_frame, text=" Export Logs", font=btn_font, bg="#ffc107", fg="black", command=self.export_logs)
        self.export_button.grid(row=7, column=0, columnspan=2, pady=5, ipadx=10)

    def detect(self):
        vehicle = self.vehicle_entry.get()
        speed = self.speed_entry.get()
        signal = self.signal_entry.get()
        msg_type = self.message_entry.get()

        if not all([vehicle, speed, signal, msg_type]):
            messagebox.showwarning("Input Error", "Please fill all fields.")
            return

        result = detect_intrusion({
            'vehicle': vehicle,
            'speed': float(speed),
            'signal_strength': float(signal),
            'message_type': msg_type
        })

        self.result_label.config(text=f"Detection Result: {result}", fg="green" if "Safe" in result else "red")
        self.last_result = (vehicle, result, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.log_button.config(state=tk.NORMAL if "Intrusion" in result else tk.DISABLED)

    def log_intrusion(self):
        vehicle, result, timestamp = self.last_result
        txid = log_to_blockchain(vehicle, timestamp, result)

        self.logs.append({
            'timestamp': timestamp,
            'vehicle': vehicle,
            'result': result,
            'txid': txid
        })

        messagebox.showinfo("Blockchain", f"Intrusion logged to blockchain. TxID: {txid}")
        self.log_button.config(state=tk.DISABLED)

    def export_logs(self):
        if not self.logs:
            messagebox.showwarning("No Logs", "No logs to export.")
            return

        df = pd.DataFrame(self.logs)
        df.to_csv("vanet_intrusion_logs.csv", index=False)
        messagebox.showinfo("Export Success", "Logs exported to vanet_intrusion_logs.csv")

if __name__ == '__main__':
    root = tk.Tk()
    app = VANETGUI(root)
    root.mainloop()