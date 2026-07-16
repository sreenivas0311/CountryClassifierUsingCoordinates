
#save model
import joblib

from CountryClassifierTrain import *

checkpoint_path = "country_classifier_checkpoint.pth"

if os.path.exists(checkpoint_path):

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("Checkpoint loaded successfully!")

    torch.save(
        model.state_dict(),
        "country_classifier_model.pth"
    )

    joblib.dump(
        preprocessor.scaler,
        "scaler.pkl"
    )

    joblib.dump(
        preprocessor.encoder,
        "label_encoder.pkl"
    )

    print("\nModel exported successfully!")
    print("Saved Files:")
    print("- country_classifier_model.pth")
    print("- scaler.pkl")
    print("- label_encoder.pkl")

else:

    print("No checkpoint found! Train the model first.")