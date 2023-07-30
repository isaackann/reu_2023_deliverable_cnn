import sys
if not '..' in sys.path: sys.path.append('..')
import argparse
import os

#-------------------------------------------------------------------------------------------------
# Get config json from terminal

parser = argparse.ArgumentParser(description="Train U-Net CNN for Sodium MRI AGR")
parser.add_argument("--config_json", default="scripts/train_config.json", help="Filename for training config json.")
parser.add_argument("--track_metrics", default=False, help="Save loss, PSNR, and SSIm at each batch.")

args = parser.parse_args()

config_json = args.config_json
track_metrics = args.track_metrics

if track_metrics and not os.path.exists('scripts/training_log'):
    os.makedirs('scripts/training_log')

#-------------------------------------------------------------------------------------------------
# Load libraries and modules

import torch
import torch.nn as nn
import torchmetrics.functional as metrics
import numpy as np
import json
import random
from torch.utils.data import DataLoader
from model_data_structures import TrainingDataset, ValidationDataset, save_to_json
from model_modules import SuperResUNetCNN
from rich.progress import Progress
from datetime import datetime

#-------------------------------------------------------------------------------------------------
# Process training config file

with open(config_json, 'r') as f:
    config = json.load(f)

num_batches = config["num_batches"]
batch_size = config["batch_size"]
patch_size = config["patch_size"]
learning_rate = config["learning_rate"]

np.random.seed(26)  # Set random seed for reproducibility

training_files = config["training_files"]
validation_files = config["validation_files"]

# Shuffle order of files
random.shuffle([i for i in range(len(training_files))])
random.shuffle([i for i in range(len(validation_files))])

#-------------------------------------------------------------------------------------------------
# Set up model for training

model = SuperResUNetCNN()
model = nn.DataParallel(model)

LOSS_FXN = nn.MSELoss()
OPTIMIZER = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Halves learning rate if loss plateaus after 160 batches (floor of 1e-4)
schedule_kwargs = dict(mode='min', factor=0.5, patience=160, min_lr=1e-4, verbose=True)
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, **schedule_kwargs)

# Device management (including multiple GPU usage)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Using {device} device')

cur_date = datetime.now().strftime("%m_%d")  # Used in model weights filename

#-------------------------------------------------------------------------------------------------
# Load data

training_dataset = TrainingDataset(training_files, batch_size=batch_size, n=patch_size)
validation_dataset = ValidationDataset(validation_files)

training_dataloader = DataLoader(training_dataset, batch_size=batch_size)
validation_dataloader = DataLoader(validation_dataset, batch_size=1)

#-------------------------------------------------------------------------------------------------
# Loops for validation and training

def train_loop(train_dataloader):
    model.train()
    
    for index, tensors in enumerate(train_dataloader):
        concat_tensor, agr_tensor = tensors[0].to(device), tensors[1].to(device)
        
        # Compute prediction and loss    
        pred = model(concat_tensor)
        loss = LOSS_FXN(pred, agr_tensor)
        
        # Backpropagation to update gradients
        loss.backward()
        OPTIMIZER.step()
        OPTIMIZER.zero_grad()

        # Display loss
        loss, current = loss.item(), (index + 1) * len(concat_tensor)
        print(f"loss: {loss:>7f}  [{current:>2d}/{len(train_dataloader.dataset):>2d}]")


""" Validation/Test Loop"""
def validation_loop(validation_dataloader):
    model.eval()
    
    val_loss = 0
    loss_list, psnr_list, ssim_list = [], [], []  # Lists to track performance
        
    with torch.no_grad():
        for tensors in validation_dataloader:
            concat_tensor, agr_tensor = tensors[0].to(device), tensors[1].to(device)

            # Calculate prediction on validation dataset
            pred = model(concat_tensor)
            
            # Calculate and save performance metrics            
            loss_item = LOSS_FXN(pred, agr_tensor).item()
            val_loss += loss_item
            
            loss_list.append(loss_item)
            psnr_list.append(metrics.peak_signal_noise_ratio(pred, agr_tensor, data_range=1).item())
            ssim_list.append(metrics.structural_similarity_index_measure(pred, agr_tensor).item())
            
    avg_loss = val_loss / len(validation_dataloader)
    print(f"\nValidation Loop: \n Avg loss: {avg_loss:>8f}")

    return avg_loss, loss_list, psnr_list, ssim_list

#-------------------------------------------------------------------------------------------------
# Run training and validation loops

avg_loss_list, losses, psnrs, ssims = [], [], [], []

with Progress() as progress:
    task = progress.add_task("[green]Training...", total=num_batches)  # Initialize progress bar

    # Train model for provided number of epochs
    for b in range(num_batches):
        print(f'\nBatch {b+1}\n-----------------------')
        train_loop(training_dataloader)
        avg_loss, loss_list, psnr_list, ssim_list = validation_loop(validation_dataloader)

        # Adjust Learning Rate if necessary
        SCHEDULER.step(avg_loss)

        # Save metrics
        avg_loss_list.append(avg_loss)
        losses.extend(loss_list)
        psnrs.extend(psnr_list)
        ssims.extend(ssim_list)
        
        if track_metrics:
            save_to_json(avg_loss_list, f'Avg Loss ({cur_date})')
            save_to_json(losses, f'Val Loss ({cur_date})')
            save_to_json(psnrs, f'Val PSNR ({cur_date})')
            save_to_json(ssims, f'Val SSIM ({cur_date})')
        
        filename = f'scripts/training_log/3d_weights_{cur_date}.pth' if track_metrics else f'scripts/3d_weights_{cur_date}.pth'
        torch.save(model.state_dict(), filename)
            
        progress.update(task, completed=b+1)  # Update progress bar
            
print(f'Model Training Session Completed!')
