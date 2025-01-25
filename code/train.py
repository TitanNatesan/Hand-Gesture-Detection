import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Custom Dataset
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the model
class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def load_gesture_data(directory):
    data = []
    labels = []
    for file_path in glob.glob(os.path.join(directory, "*.csv")):
        gesture_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            data.append(row.values)
            labels.append(gesture_name)
    return np.array(data, dtype=np.float32), np.array(labels)

def main():
    set_seed()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a directory for the model artifacts
    model_dir = "Model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Set paths
    gesture_data_dir = "gesture_data"
    model_save_path = os.path.join(model_dir, "hand_gesture_model.pth")
    label_classes_path = os.path.join(model_dir, "label_encoder_classes.npy")
    scaler_mean_path = os.path.join(model_dir, "scaler_mean.npy")
    scaler_scale_path = os.path.join(model_dir, "scaler_scale.npy")
    
    # Load all CSV files and combine into a single dataset
    print("Loading gesture data...")
    X, y = load_gesture_data(gesture_data_dir)
    print(f"Data loaded: {X.shape[0]} samples with {X.shape[1]} features each.")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save label encoder classes
    np.save(label_classes_path, label_encoder.classes_)
    
    # Split into train, validation, and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val
    )  # 0.1111 x 0.9 ~ 0.1 of original data
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save the scaler parameters
    np.save(scaler_mean_path, scaler.mean_)
    np.save(scaler_scale_path, scaler.scale_)
    
    # Create datasets and dataloaders
    train_dataset = GestureDataset(X_train, y_train)
    val_dataset = GestureDataset(X_val, y_val)
    test_dataset = GestureDataset(X_test, y_test)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize the model
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    model = GestureNet(input_size, num_classes).to(device)
    print(model)
    
    # Define loss and optimizer with weight decay for regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Early stopping parameters
    early_stopping_patience = 10
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    num_epochs = 100
    
    # Training loop with validation
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        train_loss = running_loss / len(train_dataset)
        train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
        
        val_loss = running_val_loss / len(val_dataset)
        val_acc = correct_val / total_val
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved with Val Acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epochs.")
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    
    # Load the best model for testing
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    # Testing
    print("Evaluating on test data...")
    correct_test = 0
    total_test = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = correct_test / total_test
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save training details if needed (e.g., best validation accuracy)
    np.save(os.path.join(model_dir, "best_val_acc.npy"), best_val_acc)

if __name__ == "__main__":
    main()