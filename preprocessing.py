import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 🔹 Load NSL-KDD dataset (downloaded from UNB or Kaggle)
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

# 🔹 Optional: Simplify labels (normal = 0, attack = 1)
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# 🔹 Encode categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']
encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# 🔹 Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# 🔹 Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 🔹 Save for reuse (optional)
np.savez("nslkdd_preprocessed.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

# 🔹 Print shapes
print(f"Training samples: {X_train.shape}, Testing samples: {X_test.shape}")
import matplotlib.pyplot as plt

# Count attacks vs normal
label_counts = df['label'].value_counts()
labels = ['Normal', 'Attack']
sizes = [label_counts[0], label_counts[1]]


