import torch
from torch import nn
from src.models import MyModel


def load_model(model_path: str, input_size: int, hidden_size: int = 64,
               num_hidden_layers: int = 2, output_size: int = 1):
    """Load and prepare model for inference"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = MyModel(
        input_size=input_size,
        hidden_size=hidden_size,
        number_of_hidden_layers=num_hidden_layers,
        output_size=output_size,
        dropout_p=0.0  # Disable dropout for inference
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    return model, device


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Make predictions on input data"""
    # Ensure input is on the correct device
    x = x.to(device)

    # Make prediction
    with torch.no_grad():
        predictions = model(x)

    return predictions.cpu()


def main():
    # Load model
    model_path = 'best_model.pt'
    input_size = 8  # Replace with your actual input size
    model, device = load_model(model_path, input_size)

    # Load test data
    test_data = torch.load("../data/test_data.pt", weights_only=False)

    # Get predictions
    predictions = predict(model, test_data, device)

    # Print results
    print(f"First 5 predictions:")
    print(predictions[:5].numpy().flatten())

    print(f"\nPrediction statistics:")
    pred_numpy = predictions.numpy()
    print(f"Mean: {pred_numpy.mean():.4f}")
    print(f"Std: {pred_numpy.std():.4f}")


if __name__ == "__main__":
    main()