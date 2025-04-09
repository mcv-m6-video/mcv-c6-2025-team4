#!/usr/bin/env python3
"""
File containing the main training script.
"""

#Standard imports
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

#Local imports
from util.io import load_json, store_json
from util.eval_spotting import evaluate
from dataset.datasets import get_datasets
from model.model_spotting import Model
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


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
    args.device = config['device']
    args.num_workers = config['num_workers']

    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    total_steps = args.num_epochs * num_steps_per_epoch
    warmup_steps = args.warm_up_epochs * num_steps_per_epoch
    min_lr = 0.0

    print('[LR Scheduler] Warm-up for {} epochs ({} steps)'.format(
        args.warm_up_epochs, warmup_steps
    ))

    # Lambda para warmup lineal
    def warmup_lambda(step):
        return min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0

    # Warmup (subida lineal)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Cosine annealing decay
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=min_lr
    )

    # SequentialLR: combina warmup + coseno
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    return args.num_epochs, scheduler


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
    classes, train_data, val_data, test_data,val_datav2 = get_datasets(args)

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

    # Model
    model = Model(args=args)

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})
    
    if not args.only_test:
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)

        losses = []
        best_map = -1  # mAP es mejor cuanto más alto
        epoch = 0

        patience = 5
        epochs_no_improve = 0
        # _, _, val_video_data, _ = get_datasets({**args, 'split': 'val', 'mode': 'video'})

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):
            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler
            )

            # Usamos evaluate para obtener mAP en validación
            val_map, _ = evaluate(model, val_datav2, nms_window=5)
            val_loss = model.epoch(val_loader)

            better = False
            if val_map > best_map:
                best_map = val_map
                better = True
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f'[Epoch {epoch}] Train loss: {train_loss:.5f}, Val loss: {val_loss:0.5f},  Val mAP: {val_map:.5f}')
            if better:
                print('New best mAP epoch!')

            losses.append({'epoch': epoch, 'train': train_loss, 'val_loss': val_loss, 'val_map': val_map})

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

                if better:
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt'))

                

            if epochs_no_improve >= patience:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_last.pt'))
                print(f'Early stopping: No improvement in mAP for {patience} epochs.')
                break

            

        
        model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

    print('START INFERENCE')
    # Evaluation on test split
    map_score, ap_score = evaluate(model, test_data, nms_window = 5)
    print('1')
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