import os
import datetime
import argparse
import random

import torch
import lightning

import utils.globals as uglobals
import utils.data_utils as data_utils
import utils.logging_utils as logging_utils

from models.placeholder_model import PlaceholderModel

def main(args):
    # Seeding
    lightning.seed_everything(uglobals.SEED)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    if args.force_cpu:
        device = torch.device('cpu')
        accelerator = 'cpu'
    print(f'Device: {device}')

    # Logging
    date_str = str(datetime.datetime.now())[:-7].replace(':','-').replace(' ', '_')
    logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir = f'{uglobals.RUNS_DIR}/{args.task}', 
        name=args.name, 
        version=date_str,
    )

    # Print and save args
    logging_utils.print_and_save_args_uglobals(args, logger)

    # Create model and data loaders
    if args.task == 'placeholder':
        train_loader = data_utils.get_placeholder_loader(args.batch_size)
        dev_loader = data_utils.get_placeholder_loader(args.batch_size, shuffle=False)
        model = PlaceholderModel(args)
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
        enable_progress_bar=args.debug
    )
    if args.debug:
        trainer.fast_dev_run = 5

    # Train
    trainer.fit(model, train_loader, dev_loader)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basics
    parser.add_argument('--name', type=str, default='unnamed')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')
    
    # Formulation
    parser.add_argument('--task', type=str, default='', choices=['placeholder'])

    # Training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--max_n_epochs', default=-1, type=int)
    parser.add_argument('--eval_n_epoch', default=1, type=int)
    parser.add_argument('--checkpoint', default='', type=str)

    args = parser.parse_args()
    args.uglobals = logging_utils.module_to_dict(uglobals)

    if args.debug:
        args.name = 'debug'
        # args.debug = False

        args.task = 'placeholder'
        
        args.batch_size = 3
        args.max_n_epochs = 4

    main(args)
    