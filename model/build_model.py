import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:    
    from gesture_model import GestureModel
except ImportError:
    from model.gesture_model import GestureModel

DATASET_PATH = Path("data/gestures_dataset.csv")
MODEL_SAVE_PATH = Path("model/gesture_model.pth")

BATCH_SIZE = 32
TEST_SPLIT = 0.2
RANDOM_SEED = 42
HIDDEN_UNITS = 128

if torch.cuda.is_available():
    device = torch.device("cuda") 
    print("Using GPU for training.")
else:
    device = torch.device("cpu")
    print("Using CPU for training.")


#Getting our dataset
def load_dataset():
    if DATASET_PATH.exists():
        print(f"Loading dataset from {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
    else:
        print(f"Dataset file {DATASET_PATH} not found. Please run the data collection script first.")
        exit(1)
    
    if df.empty:
        print("Dataset is empty. Please collect some gesture data before training.")
        exit(1)
    
    if df.isnull().values.any():
        print("Dataset contains null/NaN values. Please check your data collection process.")
        exit(1)
    
    if "gesture" not in df.columns:
        print("Dataset is missing 'gesture' column. Please check your data collection process.")
        exit(1)

    if "timestamp" not in df.columns:
        print("Dataset is missing 'timestamp' column. Please check your data collection process.")
        exit(1)

    print(f"Dataset loaded successfully with {len(df)} samples.")
    print(f"Sample data:\n{df.head()}")

    return df

df = load_dataset()
    

# Train test split and separate feature from target
def prepare_data(df):
    X = df.drop(columns=["gesture", "timestamp"]).values
    y = df["gesture"].values

    # Using encode as transormer to convert the string labels into integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = TEST_SPLIT, random_state = RANDOM_SEED, stratify=y_encoded)
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")
    return X_train, X_test, y_train, y_test, label_encoder

X_train, X_test, y_train, y_test, label_encoder = prepare_data(df)

# Dataloders
train_data = TensorDataset(torch.tensor(X_train, dtype = torch.float32), torch.tensor(y_train, dtype = torch.long))
test_data = TensorDataset(torch.tensor(X_test, dtype = torch.float32), torch.tensor(y_test, dtype = torch.long))

train_dataloader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(dataset = test_data, batch_size = BATCH_SIZE, shuffle = False)

# Getting the number of in features and out features for the model

in_features = X_train.shape[1]
out_features = len(label_encoder.classes_)

print(f"Number of input features: {in_features}")
print(f"Number of output classes: {out_features}")

model_1 = GestureModel(input_shape = in_features,
                      hidden_units = HIDDEN_UNITS,
                      output_shape = out_features).to(device)


# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr = 0.01)

# Training loop

def train_model():
    epochs = 100

    for epoch in range(epochs):
        model_1.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model_1(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        avg_train_loss = train_loss / len(train_dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | train_loss={avg_train_loss:.4f}")

        # Evaluation loop
        model_1.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.inference_mode():
            for X_test_batch, y_test_batch in test_dataloader:
                X_test_batch = X_test_batch.to(device)
                y_test_batch = y_test_batch.to(device)

                y_pred = model_1(X_test_batch)
                loss = loss_fn(y_pred, y_test_batch)
                test_loss += loss.item() * X_test_batch.size(0)

                preds = torch.argmax(y_pred, dim=1)
                correct += (preds == y_test_batch).sum().item()
                total += y_test_batch.size(0)

        avg_test_loss = test_loss / len(test_dataloader.dataset)
        test_accuracy = correct / total
        print(f"             | test_loss={avg_test_loss:.4f}  | test_acc={test_accuracy:.4f}")

if __name__ == "__main__":
    train_model()
    checkpoint = {
        "model_state_dict": model_1.state_dict(),
        "class_names": label_encoder.classes_.tolist(),
        "input_shape": in_features,
        "hidden_units": HIDDEN_UNITS,
        "output_shape": out_features,
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")