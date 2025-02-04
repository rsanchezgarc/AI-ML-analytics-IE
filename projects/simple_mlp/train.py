import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from src.datamanager import MyDataset, create_dataloader
from src.models import MyModel
import matplotlib.pyplot as plt
from typing import Tuple, List


def get_device() -> torch.device:
    """Automatically select the best available device"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def prepare_data(input_path: str, labels_path: str, batch_size: int = 64) -> Tuple:
    """Load and prepare data"""
    x = torch.load(input_path, weights_only=False)
    y = torch.load(labels_path, weights_only=False)

    if len(y.shape) == 1:
        y = y.view(-1, 1)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    dataset_train = MyDataset(x_train, y_train)
    dataset_val = MyDataset(x_val, y_val)

    device = get_device()
    dl_train = create_dataloader(dataset_train,
                                 batch_size=batch_size,
                                 num_workers=0,
                                 shuffle=True,
                                 use_gpu=device.type.startswith('cuda'))

    dl_val = create_dataloader(dataset_val,
                               batch_size=batch_size,
                               num_workers=0,
                               shuffle=False,
                               use_gpu=device.type.startswith('cuda'))

    return dl_train, dl_val, dataset_train, dataset_train[0][0].shape[0]


def train_epoch(
        model: nn.Module,
        dl_train: torch.utils.data.DataLoader,
        dl_val: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch and return average losses"""
    model.train()
    total_loss_train = 0
    num_batches_train = 0

    for data, target in dl_train:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        total_loss_train += loss.item()
        num_batches_train += 1

    model.eval()
    total_loss_val = 0
    num_batches_val = 0

    with torch.no_grad():
        for data, target in dl_val:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = loss_fn(out, target)
            total_loss_val += loss.item()
            num_batches_val += 1

    avg_loss_train = total_loss_train / num_batches_train
    avg_loss_val = total_loss_val / num_batches_val

    return avg_loss_train, avg_loss_val


def plot_training_curves(train_losses: List[float], val_losses: List[float]) -> None:
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Hyperparameters
    BATCH_SIZE = 64
    HIDDEN_SIZE = 64
    NUM_HIDDEN_LAYERS = 2
    OUTPUT_SIZE = 1
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    NUM_EPOCHS = 50

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Prepare data
    dl_train, dl_val, dataset_train, input_size = prepare_data(
        "../data/input_data.pt",
        "../data/labels.pt",
        BATCH_SIZE
    )

    # Initialize model
    model = MyModel(
        input_size,
        HIDDEN_SIZE,
        NUM_HIDDEN_LAYERS,
        OUTPUT_SIZE,
        dropout_p=0.5
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        loss_train, loss_val = train_epoch(
            model, dl_train, dl_val,
            optimizer, loss_fn, device
        )
        train_losses.append(loss_train)
        val_losses.append(loss_val)

        # Save best model
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            torch.save(model.state_dict(), 'best_model.pt')

        print(f"Epoch {epoch:3d}: Train Loss = {loss_train:.4f}, Val Loss = {loss_val:.4f}")

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    plot_training_curves(train_losses, val_losses)


if __name__ == "__main__":
    main()