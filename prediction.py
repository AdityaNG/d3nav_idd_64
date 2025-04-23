import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from data import create_idd_datasets
from model import D3NavIDD

# Define parameters for D3Nav model
TARGET_SIZE = 128  # Height (D3Nav expects 128x256 images)
TEMPORAL_CONTEXT = 2  # Number of input frames
NUM_LAYERS = 24  # Number of total layers in the model
USE_COMMA_GPT = True

# Load the D3Nav model
model = D3NavIDD(
    temporal_context=TEMPORAL_CONTEXT,
    num_layers=NUM_LAYERS,
    use_comma_gpt=USE_COMMA_GPT,
    attention_dropout_p=0.0,
)

# Load the checkpoint (update this path to your Lightning checkpoint)
checkpoint_path = "save_model/2025-04-23T11-27-12/model-epoch=02-val_loss=3.449869.ckpt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)

# The Lightning checkpoint has a different structure - it contains 'state_dict' with 'model.' prefix
# We need to remove the 'model.' prefix from the keys
state_dict = {}
for key in checkpoint['state_dict']:
    if key.startswith('model.'):
        # Remove the 'model.' prefix
        new_key = key[6:]  # Skip the first 6 characters ('model.')
        state_dict[new_key] = checkpoint['state_dict'][key]
    else:
        # If there's no 'model.' prefix, keep as is
        state_dict[key] = checkpoint['state_dict'][key]

# Load the processed state dict into our model
model.load_state_dict(state_dict)

# Move the model to the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to evaluation mode
model.eval()

# Prepare the data
dataset_root = "/media/NG/datasets/idd/idd_temporal_train_3"  # Path to the dataset root folder

_, val_dataset = create_idd_datasets(
    dataset_root=dataset_root,
    n_frames_input=TEMPORAL_CONTEXT,
    n_frames_output=1,
    frame_stride=5,
    target_size=TARGET_SIZE,  # D3Nav expects height=128, width=256
    train_split_ratio=0.8,
    seed=1996,  # Match the seed used in training
    motion_threshold=0.1  # Match the motion threshold from training
)

validLoader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Parameters for limiting predictions
num_batches_to_process = 128  # Number of batches to process
num_samples_per_batch = 128   # Number of samples per batch to process

# Create a directory for saving prediction images
os.makedirs('predictions', exist_ok=True)

# Make predictions and display input vs predicted vs ground truth frames
with torch.no_grad():
    for batch_idx, (idx, targetVar, inputVar, frozen, _) in enumerate(validLoader):
        if batch_idx >= num_batches_to_process:
            break  # Stop after processing the specified number of batches

        inputs = inputVar.to(device)  # Shape: [B, 2, 3, 128, 256]
        targets = targetVar.to(device)  # Shape: [B, 1, 3, 128, 256]

        # Predict the next frame
        pred, _, _ = model(inputs, targets)  # We don't need gt_reconst anymore

        # Convert to NumPy
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        pred = pred.cpu().numpy()

        # Iterate over each sample in the batch
        for sample_idx in range(min(num_samples_per_batch, pred.shape[0])):
            # Create a figure to display input, predicted, and ground truth frames in a 2x2 grid
            plt.figure(figsize=(12, 10))
            
            # Create a 2x2 grid layout
            # Row 1: Input frames
            # Row 2: Predicted frame and Ground Truth frame
            
            # Input Frame 1 (Top-Left)
            plt.subplot(2, 2, 1)
            frame = inputs[sample_idx, 0, :, :, :]  # First input frame
            frame = np.clip(frame, 0, 255) / 255.0  # Normalize from 0-255 to 0-1
            plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C)
            plt.title('Input Frame 1')
            plt.axis('off')
            
            # Input Frame 6 (Top-Right)
            plt.subplot(2, 2, 2)
            frame = inputs[sample_idx, 1, :, :, :]  # Second input frame
            frame = np.clip(frame, 0, 255) / 255.0  # Normalize from 0-255 to 0-1
            plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C)
            plt.title('Input Frame 6')
            plt.axis('off')
            
            # Predicted Frame 11 (Bottom-Left)
            plt.subplot(2, 2, 3)
            frame = pred[sample_idx, 0, :, :, :]  # Predicted frame
            frame = np.clip(frame, 0, 255) / 255.0  # Normalize from 0-255 to 0-1
            plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C)
            plt.title('Predicted Frame 11')
            plt.axis('off')
            
            # Ground Truth Frame 11 (Bottom-Right)
            plt.subplot(2, 2, 4)
            frame = targets[sample_idx, 0, :, :, :]  # Ground truth frame
            frame = np.clip(frame, 0, 255) / 255.0  # Normalize from 0-255 to 0-1
            plt.imshow(frame.transpose(1, 2, 0))  # Transpose to (H, W, C)
            plt.title('Ground Truth Frame 11')
            plt.axis('off')
            
            plt.suptitle(f'Row 1: Input Frames (1, 6) | Row 2: Predicted Frame (11), Ground Truth Frame (11)',
                         fontsize=12)
            plt.tight_layout()
            save_path = os.path.join('predictions', f'prediction_batch{batch_idx}_sample{sample_idx}.png')
            plt.savefig(save_path)
            plt.close()  # Close the figure to save memory
            
            # Only show the first sample, then just save the rest
            if batch_idx == 0 and sample_idx == 0:
                plt.show()

            # Calculate and print metrics
            # Mean Squared Error between prediction and ground truth
            mse = np.mean((pred[sample_idx, 0] - targets[sample_idx, 0])**2)
            print(f"Batch {batch_idx}, Sample {sample_idx} - MSE: {mse:.6f}")
            
            # Calculate PSNR (Peak Signal-to-Noise Ratio)
            max_pixel_value = 255.0
            psnr = 10 * np.log10((max_pixel_value**2) / mse)
            print(f"Batch {batch_idx}, Sample {sample_idx} - PSNR: {psnr:.2f} dB")