import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from data import create_idd_datasets
from model import D3NavIDD  # Updated import for the new model

# Define parameters for D3Nav model
TARGET_SIZE = 128  # Height (D3Nav expects 128x256 images)
TEMPORAL_CONTEXT = 2  # Number of input frames
NUM_UNFROZEN_LAYERS = 3  # Number of unfrozen layers (not relevant for inference)

# Load the D3Nav model
model = D3NavIDD(
    temporal_context=TEMPORAL_CONTEXT,
    num_unfrozen_layers=NUM_UNFROZEN_LAYERS
)

# Load the checkpoint (update this path to your new checkpoint)
# checkpoint_path = r'C:\Users\YASHAS\capstone\conv_idd_64\save_model\2023-05-15T00-00-00\checkpoint_best.pth.tar'
checkpoint_path = "save_model/2023-05-15T00-00-00/checkpoint_1_0.288700.pth.tar"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)

# Load the model state dict
model.load_state_dict(checkpoint['state_dict'])

# Move the model to the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Prepare the data
# dataset_root = r'C:\Users\YASHAS\capstone\baselines\conv_idd_64\idd_temporal_train_4'  # Path to the dataset root folder
dataset_root = "/media/NG/datasets/idd/idd_temporal_train_3"  # Path to the dataset root folder

_, val_dataset = create_idd_datasets(
    dataset_root=dataset_root,
    n_frames_input=TEMPORAL_CONTEXT,
    n_frames_output=1,
    frame_stride=5,
    target_size=TARGET_SIZE,  # D3Nav expects height=128, width=256
    train_split_ratio=0.8,
    seed=None
)

validLoader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Parameters for limiting predictions
num_batches_to_process = 1  # Number of batches to process
num_samples_per_batch = 1   # Number of samples per batch to process

# Make predictions and display input vs predicted vs ground truth frames
with torch.no_grad():
    for batch_idx, (idx, targetVar, inputVar, frozen, _) in enumerate(validLoader):
        if batch_idx >= num_batches_to_process:
            break  # Stop after processing the specified number of batches

        inputs = inputVar.to(device)  # Shape: [B, 2, 3, 128, 256]
        targets = targetVar.to(device)  # Shape: [B, 1, 3, 128, 256]

        # Predict the next frame
        # For inference we don't pass targets, but for visualization we include them
        # to get the ground truth reconstruction
        pred, gt_reconst, _ = model(inputs, targets)  

        # Convert to NumPy
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        pred = pred.cpu().numpy()
        gt_reconst = gt_reconst.cpu().numpy()

        # Iterate over each sample in the batch
        for sample_idx in range(min(num_samples_per_batch, pred.shape[0])):
            # Create a figure to display input, predicted, and ground truth frames
            plt.figure(figsize=(15, 10))  # Adjusted size for multiple rows and frames

            # Display the input frames (Row 1)
            for i in range(inputs.shape[1]):  # Loop over input frames (2)
                plt.subplot(4, 2, i + 1)  # 4 rows, 2 columns
                frame = inputs[sample_idx, i, :, :, :]  # RGB frame (3 channels)
                frame = np.clip(frame, 0, 255) / 255.0  # Normalize from 0-255 to 0-1
                plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C)
                plt.title(f'Input Frame {i * 5 + 1}')  # Frame 1, Frame 6 (stride 5)
                plt.axis('off')

            # Display the predicted frame (Row 2)
            plt.subplot(4, 2, 3)  # Third row, first column
            frame = pred[sample_idx, 0, :, :, :]  # Single predicted frame
            frame = np.clip(frame, 0, 255) / 255.0  # Normalize from 0-255 to 0-1
            plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C)
            plt.title('Predicted Frame 11')  # Frame 11 (5 frames after second input)
            plt.axis('off')

            # Display the ground truth frame (Row 3)
            plt.subplot(4, 2, 5)  # Third row, first column
            frame = targets[sample_idx, 0, :, :, :]  # Single ground truth frame
            frame = np.clip(frame, 0, 255) / 255.0  # Normalize from 0-255 to 0-1
            plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C)
            plt.title('Ground Truth Frame 11')  # Frame 11
            plt.axis('off')

            # Display the ground truth reconstruction (Row 4)
            plt.subplot(4, 2, 7)  # Fourth row, first column
            frame = gt_reconst[sample_idx, 0, :, :, :]  # Ground truth reconstruction
            frame = np.clip(frame, 0, 255) / 255.0  # Normalize from 0-255 to 0-1
            plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C)
            plt.title('GT Reconstruction (Encoded->Decoded)')  # Encoded then decoded ground truth
            plt.axis('off')

            plt.suptitle(f'Batch {batch_idx}, Sample {sample_idx}\n'
                         f'Row 1: Input Frames (1, 6) | Row 2: Predicted Frame (11)\n'
                         f'Row 3: Ground Truth Frame (11) | Row 4: Ground Truth Reconstruction',
                         fontsize=12)
            plt.tight_layout()
            plt.savefig(f'prediction_batch{batch_idx}_sample{sample_idx}.png')
            plt.show()

            # Optional: Calculate and print metrics
            # Mean Squared Error between prediction and ground truth
            mse = np.mean((pred[sample_idx, 0] - targets[sample_idx, 0])**2)
            print(f"MSE between prediction and ground truth: {mse:.6f}")
            
            # Mean Squared Error between ground truth reconstruction and ground truth
            mse_reconst = np.mean((gt_reconst[sample_idx, 0] - targets[sample_idx, 0])**2)
            print(f"MSE between GT reconstruction and ground truth: {mse_reconst:.6f}")