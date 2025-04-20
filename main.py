import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import D3NavIDD  # Import the new model
from data import create_idd_datasets
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse

TIMESTAMP = "2023-05-15T00-00-00"
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    default=12,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('-frames_input',
                    default=2,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=1,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=100, type=int, help='sum of epochs')
parser.add_argument('--video_path', 
                    type=str,
                    default="/media/NG/datasets/idd/idd_temporal_train_3",
                    help='Path to the input video file or extracted frames directory')
parser.add_argument('--motion_threshold',
                    default=0.1,
                    type=float,
                    help='Threshold for motion detection (fraction of pixels that must change)')
parser.add_argument('--unfrozen_layers',
                    default=3,
                    type=int,
                    help='Number of GPT layers to unfreeze for training')
parser.add_argument('--target_size',
                    default=128,
                    type=int,
                    help='Image height (width will be 2x height)')
args = parser.parse_args()

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_dir = './save_model/' + TIMESTAMP

def train():
    '''
    main function to run the training
    '''
    # Create train and validation datasets using IDDTemporalDataset
    train_dataset, val_dataset = create_idd_datasets(
        dataset_root=args.video_path,
        n_frames_input=args.frames_input,
        n_frames_output=args.frames_output,
        frame_stride=5,
        target_size=args.target_size,  # Height (width will be 2x height)
        train_split_ratio=0.8,
        seed=random_seed,
        motion_threshold=args.motion_threshold
    )

    # Create DataLoaders
    trainLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    validLoader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize D3NavIDD model
    net = D3NavIDD(
        temporal_context=args.frames_input,
        num_unfrozen_layers=args.unfrozen_layers
    )
    
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    
    # Initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # Load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                     )

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    
    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            
            optimizer.zero_grad()
            net.train()
            
            # Forward pass 
            pred, gt_reconst, loss = net(inputs, label)
            
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(filter(lambda p: p.requires_grad, net.parameters()), clip_value=10.0)
            optimizer.step()
            
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
            
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                
                # Forward pass
                pred, gt_reconst, loss = net(inputs, label)
                
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        
        # Calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)


if __name__ == "__main__":
    train()