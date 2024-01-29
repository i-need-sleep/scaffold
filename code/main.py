import os
import datetime
import argparse
import random

import torch
import lightning

import utils.globals as uglobals
import utils.data_utils as data_utils
import utils.training_utils as training_utils
import utils.logging_utils as logging_utils

from models.placeholder_model import PlaceholderModel

def main(args):
    # Seeding
    lightning.seed_everything(uglobals.SEED)

    # Device
    if not torch.cuda.is_available() or args.force_cpu:
        device = torch.device('cpu')
        accelerator = 'cpu'
    else:
        device = torch.device('cuda')
        accelerator = 'gpu'
        torch.set_float32_matmul_precision('high')
    print(f'Device: {device}')

    # Logging and checkpointing
    date_str = str(datetime.datetime.now())[:-7].replace(':','-').replace(' ', '_')
    logger_dir = f'{uglobals.RUNS_DIR}/{args.task}'
    logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir = logger_dir, 
        name=args.name, 
        version=f'{args.mode}_{date_str}',
    )
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=f'{logger_dir}/checkpoints',
        save_top_k=1,
        monitor='val/loss'
    )

    # Print and save args
    logging_utils.print_and_save_args_uglobals(args, logger)

    # Create model and data loaders
    if args.task == 'placeholder':
        train_loader = data_utils.get_placeholder_loader(args.batch_size)
        dev_loader = data_utils.get_placeholder_loader(args.batch_size, shuffle=False)
        test_loader = data_utils.get_placeholder_loader(args.batch_size, shuffle=False)
        model = training_utils.load_checkpoint_if_available(PlaceholderModel, args)
    else:
        raise NotImplementedError
    
    # Trainer
    trainer = lightning.Trainer(
        max_epochs=args.max_n_epochs, 
        check_val_every_n_epoch=args.eval_n_epoch,
        accelerator=accelerator,
        logger=logger,
        deterministic=True,
        num_sanity_val_steps=2,
        enable_progress_bar=args.debug,
        log_every_n_steps=1,
        fast_dev_run=5 if args.debug else False,
        callbacks=[checkpoint_callback]
    )

    if args.mode == 'train':
        trainer.fit(model, train_loader, dev_loader)
    elif args.mode == 'test':
        trainer.test(model, dataloaders=test_loader)
    else:
        raise NotImplementedError

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basics
    parser.add_argument('--name', type=str, default='unnamed')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')
    
    # Formulation
    parser.add_argument('--task', type=str, default=None, choices=['placeholder'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--max_n_epochs', default=-1, type=int)
    parser.add_argument('--eval_n_epoch', default=1, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)

    args = parser.parse_args()
    args.uglobals = logging_utils.module_to_dict(uglobals)

    if args.debug:
        args.name = 'debug'
        # args.debug = False

        args.task = 'placeholder'
        
        args.batch_size = 16
        args.max_n_epochs = 100
        args.lr = 1e-3

        args.mode = 'test'
        args.checkpoint = '../results/runs/placeholder/checkpoints/epoch=98-step=6237.ckpt'

    main(args)
    