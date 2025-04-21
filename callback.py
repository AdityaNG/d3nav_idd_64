import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import os
from datetime import datetime

class VisualizePredictionsCallback(Callback):
    def __init__(self, output_dir="visualizations", num_samples=4, every_n_epochs=1):
        """
        A callback to visualize predictions during training.
        
        Args:
            output_dir: Directory to save the visualizations
            num_samples: Number of samples to visualize
            every_n_epochs: How often to create visualizations (every N epochs)
        """
        super().__init__()
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate and save visualizations at the end of validation epochs"""
        # Only visualize at specified intervals
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
            
        # Create timestamp subdirectory to organize visualizations by epoch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_dir = os.path.join(self.output_dir, f"epoch_{trainer.current_epoch:03d}_{timestamp}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # Set the model to evaluation mode
        pl_module.eval()
        
        # Get a batch from validation dataloader
        val_dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_dataloader))
        idx, targetVar, inputVar, _, _ = batch

        # Move data to the appropriate device
        inputs = inputVar.to(pl_module.device)
        targets = targetVar.to(pl_module.device)
        
        # Generate predictions
        with torch.no_grad():
            # Get model predictions
            pred, gt_reconst, loss = pl_module(inputs, targets)
            
            # Log loss value
            trainer.logger.experiment.add_scalar(
                'visualization/sample_loss', 
                loss.item(), 
                global_step=trainer.global_step
            )
            
            # Convert tensors to numpy for visualization
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            pred_np = pred.cpu().numpy()
            gt_reconst_np = gt_reconst.cpu().numpy()
            
            # Visualize results for a subset of samples
            for sample_idx in range(min(self.num_samples, inputs.shape[0])):
                fig = plt.figure(figsize=(15, 10))
                
                # Display input frames (top row)
                for i in range(inputs.shape[1]):  # Usually 2 frames
                    plt.subplot(4, max(2, inputs.shape[1]), i + 1)
                    frame = inputs_np[sample_idx, i]
                    frame = np.clip(frame, 0, 255) / 255.0
                    plt.imshow(frame.transpose(1, 2, 0))
                    plt.title(f'Input Frame {i+1}')
                    plt.axis('off')
                
                # Display the prediction (second row)
                plt.subplot(4, 2, 3)
                frame = pred_np[sample_idx, 0]
                frame = np.clip(frame, 0, 255) / 255.0
                plt.imshow(frame.transpose(1, 2, 0))
                plt.title('Predicted Frame')
                plt.axis('off')
                
                # Display the ground truth (third row)
                plt.subplot(4, 2, 5)
                frame = targets_np[sample_idx, 0]
                frame = np.clip(frame, 0, 255) / 255.0
                plt.imshow(frame.transpose(1, 2, 0))
                plt.title('Ground Truth')
                plt.axis('off')
                
                # Display the ground truth reconstruction (bottom row)
                plt.subplot(4, 2, 7)
                frame = gt_reconst_np[sample_idx, 0]
                frame = np.clip(frame, 0, 255) / 255.0
                plt.imshow(frame.transpose(1, 2, 0))
                plt.title('GT Reconstruction')
                plt.axis('off')
                
                # Add metrics to the plot
                mse = np.mean((pred_np[sample_idx, 0] - targets_np[sample_idx, 0])**2)
                mse_reconst = np.mean((gt_reconst_np[sample_idx, 0] - targets_np[sample_idx, 0])**2)
                
                plt.suptitle(f'Epoch {trainer.current_epoch}, Sample {sample_idx}\n'
                             f'Prediction MSE: {mse:.2f}, Reconstruction MSE: {mse_reconst:.2f}',
                             fontsize=12)
                             
                plt.tight_layout()
                
                # Save the figure
                save_path = os.path.join(epoch_dir, f'sample_{sample_idx}.png')
                plt.savefig(save_path)
                plt.close(fig)
                
                # Also log to TensorBoard
                trainer.logger.experiment.add_figure(
                    f'predictions/sample_{sample_idx}', 
                    fig, 
                    global_step=trainer.global_step
                )
                
                # Calculate error map (difference between prediction and ground truth)
                error_map = np.abs(pred_np[sample_idx, 0] - targets_np[sample_idx, 0])
                error_fig = plt.figure(figsize=(8, 4))
                plt.imshow(np.mean(error_map, axis=0), cmap='hot')
                plt.colorbar(label='Absolute Error')
                plt.title(f'Prediction Error Map - Sample {sample_idx}')
                
                # Save error map
                error_path = os.path.join(epoch_dir, f'error_map_{sample_idx}.png')
                plt.savefig(error_path)
                plt.close(error_fig)
        
        # Set the model back to training mode
        pl_module.train()