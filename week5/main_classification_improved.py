#!/usr/bin/env python3
"""
File containing the main training script for T-DEED.
"""

# Standard imports
import argparse
import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate

# Local imports
from util.io import load_json, store_json
from util.eval_classification import evaluate
from dataset.datasets import get_datasets
from model.model_classification import Model

def get_args():
    # Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, default='first_improve.json')
    parser.add_argument('--seed', type=int, default=42)
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
    args.device = config['device']
    args.num_workers = config['num_workers']

    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print(f'Using Linear Warmup ({args.warm_up_epochs}) + Cosine Annealing LR ({cosine_epochs})')
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs)
    ])

def main(args):
    # Set seed
    print('Setting seed to:', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = 'config/' + args.model + '.json'
    config = load_json(config_path)
    args = update_args(args, config)

    # Directory for storing / reading model checkpoints
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get datasets
    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets stored! Re-run with "load" mode.')
        sys.exit()
    else:
        print('Datasets loaded correctly!')

    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers)
    
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers)

    # Model
    model = Model(args=args)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)
        
        best_criterion = float('inf')
        losses = []
        
        print('START TRAINING')
        for epoch in range(num_epochs):
            train_loss = model.epoch(train_loader, optimizer, scaler, lr_scheduler=lr_scheduler)
            val_loss = model.epoch(val_loader)

            if val_loss < best_criterion:
                best_criterion = val_loss
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pt'))

            losses.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})
            store_json(os.path.join(args.save_dir, 'loss_improved.json'), losses, pretty=True)
            print(f'[Epoch {epoch}] Train loss: {train_loss:.5f} Val loss: {val_loss:.5f}')

    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'best_model.pt')))
    ap_score = evaluate(model, test_data)

    # Print results
    print(tabulate([[cls, f"{ap*100:.2f}"] for cls, ap in zip(classes.keys(), ap_score)],
                   headers=["Class", "Average Precision"], tablefmt="grid"))
    print(tabulate([["Average", f"{np.mean(ap_score)*100:.2f}"]],
                   headers=["", "Average Precision"], tablefmt="grid"))
    print('TRAINING & INFERENCE COMPLETE')

if __name__ == '__main__':
    main(get_args())
