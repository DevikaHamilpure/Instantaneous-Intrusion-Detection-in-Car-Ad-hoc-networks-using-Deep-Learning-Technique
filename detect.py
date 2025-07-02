import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# Load your dataset from a CSV file
df = pd.read_csv('traffic_data.csv')  # Replace with your actual file path

# Convert Distance to numerical values (km)
df['Distance (km)'] = df['Distance'].apply(lambda x: float(x.replace(' km', '').replace(',', '')))

# Convert Duration to minutes (assuming 'Not available' is a missing value)
df['Duration (mins)'] = df['Duration in Traffic'].apply(lambda x: None if x == 'Not available' else int(x.replace(' mins', '')))

# Drop rows with missing values in Duration
df = df.dropna(subset=['Duration (mins)'])

# Prepare the data for the model
X = df[['Distance (km)', 'Duration (mins)']]

# Standardize the data (important for anomaly detection)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use Isolation Forest for anomaly detection
model = IsolationForest(contamination=0.2)  # Adjust contamination based on your needs
df['Anomaly'] = model.fit_predict(X_scaled)

# Map -1 (outlier) and 1 (inlier)
df['Anomaly'] = df['Anomaly'].map({1: 'Inlier', -1: 'Outlier'})

# Display the results
print(df[['Origin', 'Destination', 'Distance', 'Duration in Traffic', 'Anomaly']])

# Plot the results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Distance (km)', y='Duration (mins)', hue='Anomaly', data=df, palette={'Inlier': 'blue', 'Outlier': 'red'}, s=100)

# Add labels and title
plt.title('Anomaly Detection in Travel Data', fontsize=14)
plt.xlabel('Distance (km)', fontsize=12)
plt.ylabel('Duration (mins)', fontsize=12)
plt.legend(title='Anomaly', loc='upper left')

# Show the plot
plt.show()

# Optionally, save the results to a new CSV
df.to_csv('anomaly_detection_results.csv', index=False)
