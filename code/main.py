import os
import datetime
import argparse

import torch
import lightning

import utils.globals as uglobals
import utils.logging_utils as logging_utils
import utils.lightning_patch as lightning_patch

from utils.placeholder_dataset import make_placeholder_loader

from models.placeholder_model import PlaceholderModel

def main(args):
    # Seeding
    if not args.nondeterministic:
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
    logger_dir = f'{uglobals.RUNS_DIR}/{args.task}/{args.experiment_group}'
    logger = lightning.pytorch.loggers.TensorBoardLogger(
        save_dir = logger_dir, 
        name=args.name, 
        version=f'{args.mode}_{date_str}',
    )
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=f'{logger_dir}/{args.name}/checkpoints',
        save_last=True,
        save_top_k=1,
        monitor='val/monitor' # Minimized
    )

    # Resume from the last checkpoint
    if not args.force_restart_training and args.checkpoint is None:
        last_checkpoint = f'{logger_dir}/{args.name}/checkpoints/last.ckpt'
        if os.path.exists(last_checkpoint):
            args.checkpoint = last_checkpoint
            print(f'Resuming from the last checkpoint: {args.checkpoint}')
            
    # Print and save args
    logging_utils.print_and_save_args_uglobals(args, logger)
    
    # Create model and data loaders
    # This should be the only place to change when we add new tasks/models
    if args.task == 'placeholder':
        train_loader = make_placeholder_loader(args.batch_size, shuffle=True, single_worker=args.single_worker)
        dev_loader = make_placeholder_loader(args.batch_size, shuffle=False, single_worker=args.single_worker)
        test_loader = make_placeholder_loader(args.batch_size, shuffle=False, single_worker=args.single_worker)
        model = PlaceholderModel(vars(args))
    else:
        raise NotImplementedError
    
    # Overwriting checkpoint loading
    if args.no_stict_loading:
        model.strict_loading = False
    if args.reinit_optimizers:
        lightning_patch.skip_loading_optimizers()

    # Trainer
    trainer = lightning.Trainer(
        max_epochs=args.max_n_epochs, 
        check_val_every_n_epoch=args.eval_n_epoch,
        accelerator=accelerator,
        logger=logger,
        deterministic=not args.nondeterministic,
        num_sanity_val_steps=3,
        enable_progress_bar=args.single_worker,
        log_every_n_steps=max(len(train_loader)//5, 1) if not args.debug else 1, # Log 5 times per epoch
        callbacks=[checkpoint_callback],
        # inference_mode=False if args.mode=='predict_dev' else True, # Enable grad for reverse mel spectrogram transforms
        limit_train_batches=3 if args.debug else 1.0,
        limit_val_batches=3 if args.debug else 1.0,
        limit_test_batches=3 if args.debug else 1.0,
        limit_predict_batches= args.n_prediction_batches,
    )

    if args.mode == 'train':
        trainer.fit(model, train_loader, dev_loader, ckpt_path=args.checkpoint)
    elif args.mode == 'test':
        trainer.test(model, dataloaders=test_loader, ckpt_path=args.checkpoint)
    elif args.mode == 'test_dev':
        trainer.test(model, dataloaders=dev_loader, ckpt_path=args.checkpoint)
    elif args.mode == 'predict_dev':
        trainer.predict(model, dataloaders=dev_loader, ckpt_path=args.checkpoint, return_predictions=False)
    else:
        raise NotImplementedError
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Names
    parser.add_argument('--name', type=str, default='unnamed')
    parser.add_argument('--experiment_group', type=str, default='unnamed')

    # Checkpointing
    parser.add_argument('--force_restart_training', action='store_true') # Otherwise, automatically resume from the last checkpoint
    parser.add_argument('--no_stict_loading', action='store_true')
    parser.add_argument('--reinit_optimizers', action='store_true')

    # Debugging
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--single_worker', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--nondeterministic', action='store_true')

    # Formulation
    parser.add_argument('--task', type=str, default=None, choices=['placeholder'])
    parser.add_argument('--mode', type=str, default=None, choices=['train', 'test', 'test_dev', 'predict_dev'])

    # Training
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_n_epochs', default=100, type=int) # Set to -1 to train indefinitely. Be careful with log/output files.
    parser.add_argument('--eval_n_epoch', default=1, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)

    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--lr_scheduler_start_factor', default=1, type=float)
    parser.add_argument('--lr_scheduler_warmup_epochs', default=1, type=int)
    parser.add_argument('--lr_scheduler_end_factor', default=1, type=float)
    parser.add_argument('--lr_scheduler_anneal_epochs', default=1, type=int)

    # Prediction
    parser.add_argument('--n_prediction_batches', default=4, type=int)
    
    args = parser.parse_args()
    args.uglobals = logging_utils.module_to_dict(uglobals)

    if args.debug:
        args.name = 'ours'
        args.experiment_group = 'debug'
        args.single_worker = True
        # args.debug = False

        args.task = 'unsupervised_transcription_vq'
        args.mode = 'train'

        # args.checkpoint = '../results/runs/unsupervised_transcription_vq/ours_sanity.ckpt'
        
        args.batch_size = 4
        args.max_n_epochs = 3

        args.lr = 3e-2

    main(args)