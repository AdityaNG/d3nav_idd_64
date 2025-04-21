import os
import argparse
import numpy as np
from datetime import datetime
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from model import D3NavIDD
from data import create_idd_datasets
from callback import VisualizePredictionsCallback

torch.set_float32_matmul_precision("medium")

TIMESTAMP = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

class D3NavIDDLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Initialize model
        self.model = D3NavIDD(
            temporal_context=self.hparams.frames_input,
            num_unfrozen_layers=self.hparams.unfrozen_layers,
            num_layers=self.hparams.num_layers,
        )
    
    def forward(self, x, y=None):
        return self.model(x, y)
    
    def training_step(self, batch, batch_idx):
        idx, target, input_data, _, _ = batch
        pred, gt_reconst, loss = self(input_data, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        idx, target, input_data, _, _ = batch
        pred, gt_reconst, loss = self(input_data, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # Warmup scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=3e-4, 
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1  # Warmup for the first 10% of steps
            ),
            'interval': 'step',
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class IDDDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage=None):
        # Create train and validation datasets
        self.train_dataset, self.val_dataset = create_idd_datasets(
            dataset_root=self.hparams.video_path,
            n_frames_input=self.hparams.frames_input,
            n_frames_output=self.hparams.frames_output,
            frame_stride=5,
            target_size=self.hparams.target_size,
            train_split_ratio=0.8,
            seed=self.hparams.seed,
            motion_threshold=self.hparams.motion_threshold
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=24, type=int, help='mini-batch size per GPU')
    parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('-weight_decay', default=1e-5, type=float, help='weight decay for regularization')
    parser.add_argument('-frames_input', default=2, type=int, help='sum of input frames')
    parser.add_argument('-frames_output', default=1, type=int, help='sum of predict frames')
    parser.add_argument('-epochs', default=30, type=int, help='sum of epochs')
    parser.add_argument('--video_path', type=str, default="/media/NG/datasets/idd/idd_temporal_train_3",
                        help='Path to the input video file or extracted frames directory')
    parser.add_argument('--motion_threshold', default=0.1, type=float,
                        help='Threshold for motion detection (fraction of pixels that must change)')
    parser.add_argument('--unfrozen_layers', default=6, type=int,
                        help='Number of GPT layers to unfreeze for training')
    parser.add_argument('--num_layers', default=6, type=int, help='Number of GPT layers')
    parser.add_argument('--target_size', default=128, type=int,
                        help='Image height (width will be 2x height)')
    parser.add_argument('--seed', default=1996, type=int, help='Random seed')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers for data loading')
    parser.add_argument('--precision', default='bf16-mixed', type=str,
                        help='Precision for training (16, 32, 32-true, 64, bf16)')
    parser.add_argument('--num_nodes', default=1, type=int,
                        help='Number of nodes for distributed training')
    parser.add_argument('--devices', default=-1, type=int,
                        help='Number of GPUs to use (-1 for all available)')
    
    args = parser.parse_args()
    
    # Set random seeds
    pl.seed_everything(args.seed, workers=True)
    
    # Initialize model and data module
    model = D3NavIDDLightningModule(args)
    data_module = IDDDataModule(args)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=True,
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./save_model/{TIMESTAMP}',
        filename='model-{epoch:02d}-{val_loss:.6f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    viz_callback = VisualizePredictionsCallback(
        output_dir=f'./visualizations/{TIMESTAMP}',
        num_samples=4,  # Visualize 4 samples
        every_n_epochs=1  # Create visualizations every epoch
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Define logger
    logger = TensorBoardLogger(
        save_dir='./runs',
        name=TIMESTAMP
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[early_stopping, checkpoint_callback, lr_monitor, viz_callback],
        logger=logger,
        precision=args.precision,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy='ddp_find_unused_parameters_true' if args.devices != 1 else None,
        # strategy='ddp' if args.devices != 1 else None,
        # gradient_clip_val=10.0,
        gradient_clip_val=0.0,
        log_every_n_steps=50,
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Save loss history
    np.savetxt(
        "avg_train_losses.txt",
        np.array(trainer.callback_metrics['train_loss'].cpu())
    )
    np.savetxt(
        "avg_valid_losses.txt",
        np.array(trainer.callback_metrics['val_loss'].cpu())
    )

if __name__ == "__main__":
    main()