#!/usr/bin/env python3
"""
File containing the main training script for T-DEED.
"""

#Standard imports
import argparse
import torch
import os
import numpy as np
import random
# Commenting out unused scheduler imports
# from torch.optim.lr_scheduler import (
#     ChainedScheduler, LinearLR, CosineAnnealingLR)
# Using SequentialLR instead
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate
import json
#Local imports
from util.io import load_json, store_json
from util.eval_classification import evaluate
from dataset.datasets import get_datasets
from model.model_classification import Model

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
    args.store_dir = config['save_dir'] + '/' + "splits"
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.num_workers = config['num_workers']
    args.device=config['device']
    args.flow_dir=config['flow_dir']

    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])
# New function to create improved scheduler
def get_improved_lr_scheduler(args, optimizer, num_steps_per_epoch):
    total_steps = num_steps_per_epoch * args.num_epochs
    warmup_steps = num_steps_per_epoch * args.warm_up_epochs
    
    print('Using Improved Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, args.num_epochs - args.warm_up_epochs))
    
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Main scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.learning_rate * 0.01
    )
    
    # Combine both schedulers
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    return args.num_epochs, lr_scheduler
    

def main(args):
    # Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    # Directory for storing / reading model checkpoints
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )
        
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    # Paths
    checkpoint_path = os.path.join(ckpt_dir, 'checkpoint_last.pt')
    loss_json_path = os.path.join(args.save_dir, 'loss.json')

    # Model
    model = Model(args=args)

    # Use AdamW optimizer with weight decay instead of default optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model._model.parameters() if p.requires_grad], 
        lr=args.learning_rate,
        weight_decay=0.01
    )
    scaler = torch.cuda.amp.GradScaler()

    if not args.only_test:
        # Warmup schedule
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_improved_lr_scheduler(
            args, optimizer, num_steps_per_epoch)
        
        losses = []
        best_criterion = float('inf')
        epoch = 0
        # Check if a checkpoint and loss log exist
        if os.path.exists(checkpoint_path) and os.path.exists(loss_json_path):
            print(f"Loading checkpoint from {checkpoint_path}")

            # Load model checkpoint
            model.load(torch.load(checkpoint_path))

            # Read loss.json to find the best validation epoch
            with open(loss_json_path, 'r') as f:
                loss_data = json.load(f)
            
            if loss_data:
                # Find the epoch with the lowest validation loss
                best_epoch = min(loss_data, key=lambda x: x['val'])['epoch']
                last_epoch=14
                epoch = last_epoch + 1  # Resume from the next epoch
                print(f"Resuming training from best epoch {best_epoch} (val loss: {min(loss_data, key=lambda x: x['val'])['val']})")
            else:
                print("No previous loss data found. Training from scratch.")
        else:
            print("No checkpoint found. Training from scratch.")


        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)
            
            val_loss = model.epoch(val_loader)

            better = False
            if val_loss < best_criterion:
                best_criterion = val_loss
                better = True
            
            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
                epoch, train_loss, val_loss))
            if better:
                print('New best mAP epoch!')

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss
            })

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

                if better:
                    torch.save( model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt') )

        torch.save( model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_last.pt') )
    
    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))
    
    # Evaluacion en test split
    ap_score = evaluate(model, test_data)
    
    # Clases a excluir
    excluded_classes = {"FREE KICK", "GOAL"}
    
    # Filtrar clases y sus AP correspondientes
    filtered_ap = [ap for cls, ap in zip(classes.keys(), ap_score) if cls not in excluded_classes]
    
    # Reporte por clase en tabla
    table = [[class_name, f"{ap_score[i]*100:.2f}"] for i, class_name in enumerate(classes.keys())]
    print(tabulate(table, headers=["Class", "Average Precision"], tablefmt="grid"))
    
    # Calcular AP@12 (todas las clases) y AP@10 (sin "free kick" ni "goal")
    ap_12 = np.mean(ap_score) * 100
    ap_10 = np.mean(filtered_ap) * 100
    
    # Reportar promedios en tabla
    avg_table = [
        ["AP@12", f"{ap_12:.2f}"],
        ["AP@10 (w/o free kick & goal)", f"{ap_10:.2f}"]
    ]
    print(tabulate(avg_table, headers=["", "Average Precision"], tablefmt="grid"))
    
    print('CORRECTLY FINISHED TRAINING AND INFERENCE')

    print('LAST MODEL INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_last.pt')))
    
    # Evaluacion en test split
    ap_score = evaluate(model, test_data)
    
    # Clases a excluir
    excluded_classes = {"FREE KICK", "GOAL"}
    
    # Filtrar clases y sus AP correspondientes
    filtered_ap = [ap for cls, ap in zip(classes.keys(), ap_score) if cls not in excluded_classes]
    
    # Reporte por clase en tabla
    table = [[class_name, f"{ap_score[i]*100:.2f}"] for i, class_name in enumerate(classes.keys())]
    print(tabulate(table, headers=["Class", "Average Precision"], tablefmt="grid"))
    
    # Calcular AP@12 (todas las clases) y AP@10 (sin "free kick" ni "goal")
    ap_12 = np.mean(ap_score) * 100
    ap_10 = np.mean(filtered_ap) * 100
    
    # Reportar promedios en tabla
    avg_table = [
        ["AP@12", f"{ap_12:.2f}"],
        ["AP@10 (w/o free kick & goal)", f"{ap_10:.2f}"]
    ]
    print(tabulate(avg_table, headers=["", "Average Precision"], tablefmt="grid"))
    
    print('CORRECTLY FINISHED TRAINING AND INFERENCE')

if __name__ == '__main__':
    main(get_args())