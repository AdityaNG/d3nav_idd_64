# Training and Prediction Guide

This document provides step-by-step instructions for setting up the environment, training the model, and generating predictions.

## Environment Setup

1. **Create a Conda Environment**
   ```sh
   conda create -p venv python=3.12.8 -y
   ```
   **Note:** Ensure that Python 3.12.8 was the version i had in my case,but u can replace with your version.

2. **Activate the Environment**
   ```sh
   conda activate <path_to_venv>
   ```

3. **Install Required Packages**
   ```sh
   pip install -r requirements.txt
   ```
4. Download the pytorch package from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
   In my  case :
   ```sh
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
## Loading MovingMnist Data

To load the MovingMNIST dataset from the gzip file, use the following command:
   ```sh
   python data/mm.py
   ```

## Training the Model

To start training, run:
   ```sh
   python main.py
   ```
You can customize the number of epochs and batch size as per your requirements by modifying the respective parameters in `main.py`.

### Model Checkpoints

- The trained model is saved in the current directory in the following format:
  
  *checkpoint_<epoch>_<loss>.pth.tar*
  
- It is recommended to use a checkpoint with the lowest validation loss for better predictions.

Download checkpoint for D3NavIDD:
```
num_layers           = 24
frames_input     = 2
use_comma_gpt        = True
attention_dropout_p  = 0.1
gradient_clip_val    = 0.0
batch_size           = 8x2 (2 GPUs, Batch Size per GPU 8)
lr                   = 1e-5
weight_decay         = 1e-4
epochs               = 30
```

Checkpoint: https://drive.google.com/drive/u/0/folders/1sCRWw0AUpPQnCNkY0qRiHq71s01YM5Nx

## Generating Predictions

To generate predictions, load the desired checkpoint and run:
   ```sh
   python prediction.py
   ```
![](https://github.com/Yashas-naidu/ConvLSTM/blob/main/images/movingmnist.png)
## Viewing Logs in TensorBoard

Logs for **Epoch vs Loss** are stored inside the `runs` directory. To visualize them using TensorBoard, execute:
   ```sh
   tensorboard --logdir=./runs
   ```

This will launch TensorBoard, where you can analyze the training progress visually.

---

**Note:** Ensure that all dependencies are correctly installed before proceeding with training and prediction.

