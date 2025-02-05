import optuna
from train_with_pytorch_lightning import *
from sklearn.model_selection import KFold

def objective(trial):
    hidden_size = trial.suggest_int('hidden_size', 32, 512)
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gradient_clip_val = trial.suggest_loguniform('gradient_clip_val', 0.1, 5.0)
    accumulate_grad_batches = trial.suggest_int('accumulate_grad_batches', 1, 4)

    output_dir = Path("outputs")

    x = torch.load("AI-ML-analytics-IE/projects/simple_mlp/data/input_data.pt", weights_only=False)
    y = torch.load("AI-ML-analytics-IE/projects/simple_mlp/data/labels.pt", weights_only=False)
    dataset = MyDataset(x, y)
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    val_losses = []
    
    for train_idx, val_idx in kf.split(dataset):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
    
        train_loader = create_dataloader(
            train_subset, batch_size=batch_size, num_workers=0, use_gpu=torch.cuda.is_available()
        )
        val_loader = create_dataloader(
            val_subset, batch_size=batch_size, num_workers=0, use_gpu=torch.cuda.is_available(), shuffle=False
        )
    
        model = MyModelPl(
            input_size=len(dataset[0][0]),
            hidden_size=hidden_size,
            number_of_hidden_layers=num_hidden_layers,
            output_size=1,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
    
        callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5, verbose=True, mode='min'),
            ModelCheckpoint(dirpath=output_dir, filename='model-{epoch:02d}-{val_loss:.2f}', monitor='val_loss', save_top_k=3, mode='min')
        ]
    
        logger = TensorBoardLogger(save_dir=output_dir / "logs")
    
        trainer = pl.Trainer(
            max_epochs=50, 
            callbacks=callbacks,
            logger=logger,
            accelerator='auto',
            devices=1,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val
        )
    
        # Train model
        trainer.fit(model, train_loader, val_loader)
        val_losses.append(trainer.callback_metrics['val_loss'].item())
    
    return sum(val_losses) / len(val_losses) 

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    print("Best hyperparameters:", study.best_params)

if __name__ == "__main__":
    main()