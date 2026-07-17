
from CountryClassifierTrain import *

checkpoint_path = "saved_models/country_classifier_checkpoint.pth"

if os.path.exists(checkpoint_path):

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("Checkpoint loaded successfully!")

    evaluate_model(
        model,
        preprocessor.test_loader,
        criterion,
        preprocessor.encoder
    )

else:

    print("No checkpoint found! Train the model first.")
