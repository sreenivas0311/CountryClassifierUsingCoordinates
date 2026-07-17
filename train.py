
from CountryClassifierTrain import *

checkpoint_path = "saved_models/country_classifier_checkpoint.pth"

if os.path.exists(checkpoint_path):

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("Checkpoint loaded successfully!")

train_model(
    model,
    preprocessor.train_loader,
    criterion,
    optimizer,
    epochs=30
)
