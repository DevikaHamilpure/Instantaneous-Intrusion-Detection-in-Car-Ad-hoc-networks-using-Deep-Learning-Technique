import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np


# Step 1: Load and Preprocess the Data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Assume 'label' is the target
    X = data.drop('label', axis=1)
    y = data['label']

    # Encode labels (Normal = 0, Attack = 1)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


# Step 2: Define RNN Model
class RNN_IDS(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_IDS, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden states
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)

        # Only take output from the last time step
        out = self.fc(out[:, -1, :])
        return out


# Step 3: Training Function
def train_model(model, criterion, optimizer, train_loader, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')


# Step 4: Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Accuracy Score:", accuracy_score(y_true, y_pred))


# Step 5: Main
if __name__ == "__main__":
    # Load dataset
    file_path = 'NSLKDD.csv'  # <-- Change this!
    X, y = load_data(file_path)

    # Reshape input for RNN (samples, sequence_length, input_size)
    X = np.expand_dims(X, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Hyperparameters
    input_size = X_train.shape[2]
    hidden_size = 64
    num_layers = 2
    num_classes = len(np.unique(y))
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RNN_IDS(input_size, hidden_size, num_layers, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and Evaluate
    train_model(model, criterion, optimizer, train_loader, num_epochs=20)
    evaluate_model(model, test_loader)
