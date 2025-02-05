import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
from typing import Tuple, Dict, Any
import json
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

# from customLosses import WeightedMSELoss
from datamanager import MyDataset, create_dataloader
from models import MyModel


class MyModelPl(pl.LightningModule):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            number_of_hidden_layers: int,
            output_size: int,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model architecture
        self.inner_model = MyModel(
            input_size,
            hidden_size,
            number_of_hidden_layers,
            output_size
        )

        # Loss function
        self.loss_fn = nn.HuberLoss()

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_model(x)

    @staticmethod
    def pearson_corr(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate Pearson correlation coefficient"""
        vx = y_true - torch.mean(y_true)
        vy = y_pred - torch.mean(y_pred)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """Shared step for training and validation"""
        x, y = batch
        y_hat = self(x)

        # Ensure consistent shapes
        y = y.view(-1)
        y_hat = y_hat.view(-1)

        # Calculate metrics
        loss = self.loss_fn(y_hat, y)
        corr = self.pearson_corr(y, y_hat)

        # Log metrics
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_corr', corr, on_step=True, on_epoch=True, prog_bar=True)

        return {'loss': loss, f'{stage}_corr': corr}

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._shared_step(batch, 'val')

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        x, y = batch
        y_hat = self(x)
        return {'y': y, 'y_pred': y_hat}

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


def get_predictions(model: pl.LightningModule, dataloader: torch.utils.data.DataLoader) -> Tuple[
    np.ndarray, np.ndarray]:
    """Get predictions for entire dataset"""
    trainer = pl.Trainer(accelerator='auto', devices=1)
    predictions = trainer.predict(model, dataloaders=dataloader)

    # Concatenate all predictions and actual values
    y_pred = torch.cat([batch['y_pred'] for batch in predictions]).cpu().numpy()
    y_true = torch.cat([batch['y'] for batch in predictions]).cpu().numpy()

    return y_pred.squeeze(), y_true.squeeze()


def plot_predictions(train_preds: np.ndarray, train_actuals: np.ndarray,
                     val_preds: np.ndarray, val_actuals: np.ndarray,
                     save_path: str = None) -> None:
    """Create scatter plots for training and validation predictions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Training set plot
    ax1.scatter(train_actuals, train_preds, alpha=0.5, s=10)
    ax1.plot([train_actuals.min(), train_actuals.max()],
             [train_actuals.min(), train_actuals.max()],
             'r--', lw=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'Training Set (r={np.corrcoef(train_actuals, train_preds)[0, 1]:.3f})')

    # Validation set plot
    ax2.scatter(val_actuals, val_preds, alpha=0.5, s=10)
    ax2.plot([val_actuals.min(), val_actuals.max()],
             [val_actuals.min(), val_actuals.max()],
             'r--', lw=2)
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title(f'Validation Set (r={np.corrcoef(val_actuals, val_preds)[0, 1]:.3f})')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def create_stratified_split(dataset, n_bins, n_splits):
    y = dataset.y.numpy()
    
    binned_y = np.digitize(y, bins=np.linspace(y.min(), y.max(), n_bins))
    
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
    
    for train_index, val_index in sss.split(np.zeros(len(y)), binned_y):
        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, val_index)
        return train_dataset, val_dataset 


def main():
    # Initial Configuration
    # CONFIG = {
    #     'batch_size': 256,
    #     'hidden_size': 512,
    #     'num_hidden_layers': 8,
    #     'learning_rate': 1e-4,
    #     'weight_decay': 1e-3,
    #     'max_epochs': 10,
    #     'patience': 50,
    #     'accumulate_grad_batches': 2,
    #     'gradient_clip_val': 4,
    # }

    # Optimized Configuration
    CONFIG = json.load(open("AI-ML-analytics-IE/projects/simple_mlp/src/config/best_params.json"))

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Load data
    x = torch.load("AI-ML-analytics-IE/projects/simple_mlp/data/input_data.pt", weights_only=False)
    y = torch.load("AI-ML-analytics-IE/projects/simple_mlp/data/labels.pt", weights_only=False)
    dataset = MyDataset(x, y)

    # Create stratified split
    train_dataset, val_dataset = create_stratified_split(dataset, n_bins=10, n_splits=5)

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        num_workers=0,
        use_gpu=torch.cuda.is_available()
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        num_workers=0,
        use_gpu=torch.cuda.is_available(),
        shuffle=False
    )

    # Initialize model
    model = MyModelPl(
        input_size=len(dataset[0][0]),
        hidden_size=CONFIG['hidden_size'],
        number_of_hidden_layers=CONFIG['num_hidden_layers'],
        output_size=1,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Configure callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,
            patience=CONFIG['patience'],
            verbose=True,
            mode='min'
        ),
        ModelCheckpoint(
            dirpath=output_dir,
            filename='model-{epoch:02d}-{val_loss:.2f}-{val_corr:.2f}',
            monitor='val_loss',
            save_top_k=3,
            mode='min'
        )
    ]

    # Configure logger
    logger = TensorBoardLogger(save_dir=output_dir / "logs")
    #While training (and after), you can see your logs using tensorboard --logdir outputs/logs

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG['max_epochs'],
        callbacks=callbacks,
        logger=logger,
        accelerator='auto',
        devices=1,
        accumulate_grad_batches=CONFIG['accumulate_grad_batches'],
        gradient_clip_val=CONFIG['gradient_clip_val']
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Generate and plot predictions
    train_preds, train_actuals = get_predictions(model, train_loader)
    val_preds, val_actuals = get_predictions(model, val_loader)

    plot_predictions(
        train_preds, train_actuals,
        val_preds, val_actuals,
        save_path=output_dir / "predictions.png"
    )

    # Print final metrics
    print("\nFinal Metrics:")
    print(f"Training Correlation: {np.corrcoef(train_preds, train_actuals)[0, 1]:.4f}")
    print(f"Validation Correlation: {np.corrcoef(val_preds, val_actuals)[0, 1]:.4f}")


if __name__ == "__main__":
    main()