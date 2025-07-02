import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Reshape, SimpleRNN
import tensorflow as tf

# -------------------- Data Preprocessing --------------------

# Load dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

df = pd.read_csv("NSLKDD.csv", header=0)
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Encode categorical
categorical_cols = ['protocol_type', 'service', 'flag']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Split features/label
X = df.drop('label', axis=1)
y = df['label']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape for CNN/RNN models
X_train_dl = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_dl = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# -------------------- CNN-LSTM Model --------------------
cnn_lstm_model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

cnn_lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_lstm_model.summary()

cnn_lstm_model.fit(X_train_dl, y_train, epochs=5, batch_size=128, validation_data=(X_test_dl, y_test))

# -------------------- RNN Model --------------------
rnn_model = Sequential([
    SimpleRNN(64, input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.summary()

rnn_model.fit(X_train_dl, y_train, epochs=5,batch_size=128, validation_data=(X_test_dl, y_test))

# -------------------- Evaluation --------------------
print("\nCNN-LSTM Results:")
y_pred_cnn = (cnn_lstm_model.predict(X_test_dl) > 0.5).astype("int32")
print(classification_report(y_test, y_pred_cnn))

print("\nRNN Results:")
y_pred_rnn = (rnn_model.predict(X_test_dl) > 0.5).astype("int32")
print(classification_report(y_test, y_pred_rnn))

# -------------------- Save Models --------------------
cnn_lstm_model.save("vanet_cnn_lstm_model.h5")
rnn_model.save("vanet_rnn_model.h5")
