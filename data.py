import pickle
import hashlib
import cv2
import numpy as np
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import random
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

class IDDTemporalDataset(data.Dataset):
    def __init__(self, sequence_folders, n_frames_input=2, n_frames_output=1, 
                 frame_stride=5, target_size=256, transform=None, include_sequence_info=False,
                 motion_threshold=0.01):
        '''
        Dataset for IDD temporal data with colored (RGB) frames, filtering sequences with motion
        
        Args:
            sequence_folders: list of paths to sequence folders (e.g., paths to *_leftImg8bit folders)
            n_frames_input: number of input frames (now 2)
            n_frames_output: number of output frames to predict (now 1)
            frame_stride: number of frames between consecutive input/output frames (now 5)
            target_size: size to resize frames to (square dimensions for the model)
            transform: optional additional transformations
            include_sequence_info: whether to include sequence information in output
            motion_threshold: threshold for motion detection (fraction of pixels that must change)
        '''
        super(IDDTemporalDataset, self).__init__()
        
        self.sequence_folders = sequence_folders
        self.n_frames_input = n_frames_input  # Set to 2
        self.n_frames_output = n_frames_output  # Set to 1
        self.frame_stride = frame_stride  # Set to 5
        self.n_frames_total = (n_frames_input - 1) * frame_stride + frame_stride + 1  # Total span: 10 frames (0, 5, 10)
        self.target_size = target_size
        self.transform = transform
        self.include_sequence_info = include_sequence_info
        self.motion_threshold = motion_threshold  # Threshold for motion detection
        
        # Process sequences and build index mapping
        self.sequences = self.process_sequences()
        self.index_mapping = self.create_index_mapping()
        self.length = len(self.index_mapping)
    
    def detect_motion(self, frames):
        """Detect motion between frames by calculating pixel differences"""
        # Resize frames to a smaller size for faster computation
        small_size = (64, 64)  # Smaller size to speed up computation
        total_pixels = small_size[0] * small_size[1]
        
        # Load and preprocess the first frame
        prev_frame = cv2.imread(frames[0])
        if prev_frame is None:
            return False
        prev_frame = cv2.resize(prev_frame, small_size)
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for simplicity
        
        # Compare consecutive frames
        for i in range(1, len(frames)):
            curr_frame = cv2.imread(frames[i])
            if curr_frame is None:
                return False
            curr_frame = cv2.resize(curr_frame, small_size)
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Compute absolute difference
            diff = cv2.absdiff(prev_frame, curr_frame)
            # Threshold the difference (e.g., consider a pixel changed if difference > 30)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            # Calculate the fraction of changed pixels
            changed_pixels = np.sum(thresh) / 255.0
            fraction_changed = changed_pixels / total_pixels
            
            # If enough pixels have changed, consider this sequence as having motion
            if fraction_changed > self.motion_threshold:
                return True
            
            prev_frame = curr_frame
        
        return False
    
    def process_sequences(self):
        """Process all sequence folders and identify valid frames with motion"""
        # Create cache directory if it doesn't exist
        cache_dir = os.path.expanduser("~/.d3nav_data")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a cache ID based on sequence folders and motion threshold
        sequence_hash = hashlib.md5(str(self.sequence_folders).encode()).hexdigest()
        motion_threshold_str = str(self.motion_threshold)
        cache_id = f"{sequence_hash}_{motion_threshold_str}"
        cache_file = os.path.join(cache_dir, f"sequences_cache_{cache_id}.pkl")
        
        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                print(f"Loading sequences from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Failed to load cache: {e}")
        
        # If cache doesn't exist or loading failed, process the sequences
        sequences = []
        
        for seq_folder in tqdm(self.sequence_folders, desc="process_sequences"):
            # Extract category and sequence IDs from folder path
            parts = seq_folder.split(os.sep)
            category_id = parts[-2]  # e.g., "00163"
            sequence_id = parts[-1]  # e.g., "155270_leftImg8bit"
            
            # Find all jpeg files in the sequence folder
            frame_files = sorted(glob.glob(os.path.join(seq_folder, "*.jpeg")))
            
            # Only include sequences with enough frames (at least 10 frames for stride=5)
            if len(frame_files) >= self.n_frames_total:
                # Check for motion in the sequence
                if self.detect_motion(frame_files):
                    sequences.append({
                        'folder': seq_folder,
                        'frames': frame_files,
                        'category': category_id,
                        'sequence': sequence_id,
                        'name': f"{category_id}_{sequence_id}"
                    })
                else:
                    # print(f"Skipping sequence {seq_folder} due to insufficient motion")
                    pass
            else:
                print(f"Warning: Skipping sequence {seq_folder} with only {len(frame_files)} frames (need {self.n_frames_total})")
        
        print(f"Found {len(sequences)} valid sequences with {self.n_frames_total}+ frames and sufficient motion")
        
        # Save the results to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(sequences, f)
            print(f"Cached sequences to: {cache_file}")
        except Exception as e:
            print(f"Failed to write cache: {e}")
            
        return sequences

    def create_index_mapping(self):
        """Create a mapping from dataset indices to (sequence_idx, frame_idx)"""
        index_mapping = []
        
        for seq_idx, sequence in enumerate(self.sequences):
            n_frames = len(sequence['frames'])
            
            # Add indices for all valid frame sequences with stride
            for frame_idx in range(n_frames - self.n_frames_total + 1):
                index_mapping.append((seq_idx, frame_idx))
        
        print(f"Created {len(index_mapping)} valid frame sequences for training/validation")
        return index_mapping


    def preprocess_frame(self, frame):
        """Process a frame in RGB format for D3Nav model
        
        D3Nav expects images with a height of 128 and width of 256
        """
        # Convert BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to target size (D3Nav expects height=128, width=256)
        # If target_size is set differently, we'll keep the aspect ratio 1:2
        height = self.target_size  # Default is now 128
        width = height * 2  # Default is now 256
        frame = cv2.resize(frame, (width, height))
        
        # Normalize to 0-1 range (or 0-255 as needed by D3Nav)
        # D3Nav might work better with 0-255 range values, so change as needed
        frame = frame.astype(np.float32)  # Keep as 0-255 range
        
        return frame
    
    def __getitem__(self, idx):
        """Get sequence of RGB frames starting at idx with stride"""
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.length}")
        
        # Get sequence and frame indices
        seq_idx, frame_idx = self.index_mapping[idx]
        sequence = self.sequences[seq_idx]
        
        # Get frames with stride (e.g., frame_idx, frame_idx+5, frame_idx+10)
        frames = []
        for i in range(self.n_frames_input + self.n_frames_output):  # 2 inputs + 1 output
            frame_pos = frame_idx + i * self.frame_stride  # Stride of 5
            frame_path = sequence['frames'][frame_pos]
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Failed to load image: {frame_path}")
            frame = self.preprocess_frame(frame)
            frames.append(frame)
        
        sequence_frames = np.array(frames)
        
        # Split into input and output
        input_frames = sequence_frames[:self.n_frames_input]  # First 2 frames (e.g., 0, 5)
        output_frames = sequence_frames[self.n_frames_input:]  # Last frame (e.g., 10)
        
        # Get the last input frame as "frozen" reference
        frozen = input_frames[-1].copy()
        
        # Convert to torch tensors and move channel dimension
        # Move channel dim: (seq_len, height, width, 3) -> (seq_len, 3, height, width)
        input_tensor = torch.from_numpy(input_frames).permute(0, 3, 1, 2).contiguous().float()
        output_tensor = torch.from_numpy(output_frames).permute(0, 3, 1, 2).contiguous().float()
        
        # Extra info for debugging/tracking
        if self.include_sequence_info:
            info = {
                'sequence_name': sequence['name'],
                'category': sequence['category'],
                'sequence_id': sequence['sequence'],
                'frame_idx': frame_idx,
                'frame_paths': [sequence['frames'][frame_idx + i * self.frame_stride] 
                                for i in range(self.n_frames_input + self.n_frames_output)]
            }
            return [idx, output_tensor, input_tensor, frozen, info]
        
        # Format to match the original structure
        # [idx, output_frames, input_frames, frozen, _]
        return [idx, output_tensor, input_tensor, frozen, np.zeros(1)]
    
    def __len__(self):
        return self.length

def find_all_sequence_folders(dataset_root):
    """Find all sequence folders in the dataset"""
    sequence_folders = []
    
    # Walk through the dataset structure
    for category_folder in os.listdir(dataset_root):
        category_path = os.path.join(dataset_root, category_folder)
        
        if os.path.isdir(category_path):
            # For each category, find all sequence folders
            for sequence_folder in os.listdir(category_path):
                if sequence_folder.endswith('_leftImg8bit'):
                    sequence_path = os.path.join(category_path, sequence_folder)
                    sequence_folders.append(sequence_path)
    
    print(f"Found {len(sequence_folders)} total sequence folders")
    return sequence_folders

def create_idd_datasets(
    dataset_root,
    n_frames_input=2,  # Default 2 frames input
    n_frames_output=1,  # Default 1 frame output
    frame_stride=5,  # Default stride of 5
    target_size=128,  # Default height 128 (width will be 256)
    train_split_ratio=0.8,
    seed=None,
    motion_threshold=0.01  # Default motion threshold
):
    """Create train and validation datasets from IDD temporal data with motion filtering
    
    For D3Nav model, target_size should typically be 128 (height) with width 256
    """

    # Set random seed for reproducibility
    random.seed(seed)
    
    # Find all sequence folders
    all_sequence_folders = find_all_sequence_folders(dataset_root)
    
    # Shuffle and split into train and validation sets
    random.shuffle(all_sequence_folders)
    split_idx = int(len(all_sequence_folders) * train_split_ratio)
    
    train_sequences = all_sequence_folders[:split_idx]
    val_sequences = all_sequence_folders[split_idx:]
    
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    
    # Create train and validation datasets with motion filtering
    train_dataset = IDDTemporalDataset(
        sequence_folders=train_sequences,
        n_frames_input=n_frames_input,
        n_frames_output=n_frames_output,
        frame_stride=frame_stride,
        target_size=target_size,
        motion_threshold=motion_threshold,
        include_sequence_info=True  # Enable to get frame paths for visualization
    )
    
    val_dataset = IDDTemporalDataset(
        sequence_folders=val_sequences,
        n_frames_input=n_frames_input,
        n_frames_output=n_frames_output,
        frame_stride=frame_stride,
        target_size=target_size,
        motion_threshold=motion_threshold,
        include_sequence_info=True  # Enable to get frame paths for visualization
    )
    
    return train_dataset, val_dataset

def visualize_samples(dataset, num_samples=10):
    """Visualize a few sample sequences from the dataset"""
    if len(dataset) == 0:
        print("Dataset is empty after motion filtering. Try adjusting the motion threshold.")
        return
    
    # Randomly select a few indices to visualize
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in indices:
        # Get the sample
        sample = dataset[idx]
        _, output_tensor, input_tensor, _, info = sample
        
        # Convert tensors back to numpy for visualization
        input_frames = input_tensor.numpy().transpose(0, 2, 3, 1)  # (seq_len, 3, H, W) -> (seq_len, H, W, 3)
        output_frames = output_tensor.numpy().transpose(0, 2, 3, 1)  # (seq_len, 3, H, W) -> (seq_len, H, W, 3)
        
        # Create a figure for this sample
        fig, axes = plt.subplots(1, dataset.n_frames_input + dataset.n_frames_output, figsize=(15, 5))
        fig.suptitle(f"Sample {idx} - Sequence: {info['sequence_name']}, Frame Index: {info['frame_idx']}")
        
        # Plot input frames
        for i in range(dataset.n_frames_input):
            axes[i].imshow(input_frames[i])
            axes[i].set_title(f"Input Frame {(i * dataset.frame_stride)}")
            axes[i].axis('off')
        
        # Plot output frame
        for i in range(dataset.n_frames_output):
            axes[dataset.n_frames_input + i].imshow(output_frames[i])
            axes[dataset.n_frames_input + i].set_title(f"Output Frame {(dataset.n_frames_input + i) * dataset.frame_stride}")
            axes[dataset.n_frames_input + i].axis('off')
        
        plt.show()

# Main execution block
if __name__ == '__main__':
    # Example usage
    dataset_root = r"C:\Users\YASHAS\capstone\baselines\conv_idd_64\idd_temporal_train_4"  # Path to your dataset
    
    # Create the datasets with motion filtering
    train_dataset, val_dataset = create_idd_datasets(
        dataset_root=dataset_root,
        n_frames_input=2,  # 2 input frames
        n_frames_output=1,  # 1 output frame
        frame_stride=5,  # 5 frames apart
        target_size=256,
        train_split_ratio=0.8,
        motion_threshold=0.1  # Adjust this threshold as needed
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Visualize a few samples from the training dataset
    print("Visualizing samples from the training dataset...")
    visualize_samples(train_dataset, num_samples=10)

    # Create DataLoaders for train and validation
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,          # Adjust based on your GPU memory
        shuffle=True,
        num_workers=4,         # For parallel loading
        pin_memory=True        # Speeds up transfer to GPU
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,          # Adjust based on your GPU memory
        shuffle=False,
        num_workers=4,         # For parallel loading
        pin_memory=True        # Speeds up transfer to GPU
    )