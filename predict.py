import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the neural network
class CountryClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 226)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.output_layer(x)

        return x


# Load scaler and label encoder
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# Load model
model = CountryClassifier().to(device)

model.load_state_dict(
    torch.load(
        "country_classifier_model.pth",
        map_location=device
    )
)

model.eval()


while True:
    # Get user input
    latitude = float(input("Enter Latitude : "))
    longitude = float(input("Enter Longitude: "))


    # Prepare input
    coordinates = pd.DataFrame(
        [[latitude, longitude]],
        columns=["latitude", "longitude"]
    )

    coordinates = scaler.transform(coordinates)

    coordinates = torch.tensor(
        coordinates,
        dtype=torch.float32,
        device=device
    )


    # Predict
    with torch.no_grad():

        logits = model(coordinates)

        probabilities = torch.softmax(logits, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)


    predicted_country = label_encoder.inverse_transform(
        prediction.cpu().numpy()
    )[0]


    print("\nPrediction")
    print(f"Country    : {predicted_country}")
    print(f"Confidence : {confidence.item() * 100:.2f}%")

    choice=input("want to predict more Y/N ?  \n").strip().lower()

    if choice == "n":
        break
    elif choice != "y" or choice != "n" :
        print("Type y or n")
        
