import os
# Data handling
import pandas as pd
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# PyTorch Dataset and DataLoader
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report

# Select the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Dataset file
DATASET_PATH = "country_coordinate_data.csv"

#process dataset
class Preprocessor:
    # Initialize the preprocessor
    def __init__(self):
        #laod the dataset
        self.df=pd.read_csv(DATASET_PATH)

    def display(self):
        #display dataset
        print(self.df.head(5))
        print(self.df.columns)
        print(self.df.shape)
        print(self.df["latitude"].dtype)
        print(self.df['country_name'].nunique())

    def separate_features(self):
        #separating ip features and target variable
        self.X = self.df.drop(columns={"country_name"})
        self.Y = self.df["country_name"]
    # Encode target labels
    def encode_labels(self):
        self.encoder = LabelEncoder()
        self.Y = self.encoder.fit_transform(self.Y)
    # Split the dataset into training and testing sets
    def split_data(self):
       self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, 
                                                                               test_size=0.2, 
                                                                               random_state=42,
                                                                               shuffle=True, 
                                                                               stratify=self.Y)
    
    #normalizing data
    def normalize_data(self):
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test  = self.scaler.transform(self.X_test)
    

    #converting arrays to tensors
    def convert_to_tensors(self):
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32 , device=device)
        self.X_test  = torch.tensor(self.X_test,  dtype=torch.float32 , device=device)
        self.Y_train = torch.tensor(self.Y_train, dtype=torch.int64   , device=device)
        self.Y_test  = torch.tensor(self.Y_test,  dtype=torch.int64   , device=device)

    # Create PyTorch datasets
    def create_dataset(self):
        self.train_dataset = TensorDataset(self.X_train, self.Y_train)
        self.test_dataset  = TensorDataset(self.X_test, self.Y_test)


    # Create PyTorch data loaders
    def create_dataloader(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=128, 
            shuffle=False
        )

    
        

# Define the neural network
class CountryClassifier(nn.Module):

    # Initialize the neural network
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,128) #fc-->fully connected
        self.fc2 = nn.Linear(128,128)
        self.output_layer =nn.Linear(128,226)


    # Define how data flows through the network
    def forward(self, x): #x --> batchsize
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
         
        return x 


# Create the model
model = CountryClassifier().to(device)

# Create the loss function
criterion = nn.CrossEntropyLoss()
#cerate optimizer 
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Train the model
def train_model(model, train_loader, criterion, optimizer, epochs):

    model.train()

    for epoch in range(epochs):


        epoch_loss = 0

        for X_batch, Y_batch in train_loader:

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(logits, Y_batch)

            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()

        average_loss = epoch_loss / len(train_loader)

        # Print at 0% and then every 10%
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:

            progress = (epoch / epochs) * 100 if epoch == 0 else ((epoch + 1) / epochs) * 100

            print(
                f"Epoch: {epoch+1}/{epochs} | "
                f"{progress:.1f}% | "
                f"Average Loss for epoch : {average_loss:.4f}"
            )

            #storing output in file
            with open("training_log.txt", "a") as file:
                file.write(
                    f"Epoch: {epoch + 1}/{epochs} | "
                    f"Loss: {average_loss:.4f}\n"
                )
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },"saved_models/country_classifier_checkpoint.pth")

    
    with open("training_log.txt", "a") as file:
        file.write("Checkpoint Saved\n\n")

# Test the model
def evaluate_model(model, test_loader, criterion, encoder):

    # Set model to evaluation mode
    model.eval()

    test_loss = 0
    correct_predictions = 0
    total_predictions = 0

    all_predictions = []
    all_labels = []

    # Disable gradient calculation
    with torch.no_grad():

        for X_batch, Y_batch in test_loader:

            # Forward pass
            logits = model(X_batch)

            # Calculate loss
            loss = criterion(logits, Y_batch)

            test_loss += loss.item()

            # Get predicted class
            predictions = torch.argmax(logits, dim=1)

            # Count correct predictions
            correct_predictions += (predictions == Y_batch).sum().item()

            # Count total predictions
            total_predictions += Y_batch.size(0)

            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(Y_batch.cpu().numpy())

    # Calculate average test loss
    average_test_loss = test_loss / len(test_loader)

    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100

    print(f"Test Loss: {average_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%\n")

    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=encoder.classes_,
            digits=4,
            zero_division=0
        )
    )

    # Store test results in log file
    with open("training_log.txt", "a") as file:
        file.write(
            f"Test Loss: {average_test_loss:.4f}\n"
            f"Test Accuracy: {accuracy:.2f}%\n\n"
        )

    return average_test_loss, accuracy

    # Create preprocessor object
preprocessor = Preprocessor()

# Run preprocessing pipeline
preprocessor.separate_features()
preprocessor.encode_labels()
preprocessor.split_data()
preprocessor.normalize_data()
preprocessor.convert_to_tensors()
preprocessor.create_dataset()
preprocessor.create_dataloader()

