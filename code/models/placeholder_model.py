from pathlib import Path
import itertools

import torch
import lightning
from lightning.pytorch.utilities import grad_norm

class PlaceholderModel(lightning.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )

        # Initialize the output folder
        self.output_folder = f'{args["uglobals"]["OUTPUTS_DIR"]}/{args["task"]}/{args["experiment_group"]}/{args["name"]}'
        if args['mode'] == 'predict_dev':
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    # Optimization
    def configure_optimizers(self):
        params = [self.model.parameters()]
        self.params_to_update = itertools.chain(*params)
        optimizer = torch.optim.Adam(self.params_to_update, lr=self.args['lr'])
        # Since Adam is per-parameter, we don't need to re-initalize the optimizer when switching training modes

        # LR scheduler
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=self.args['lr_scheduler_start_factor'], end_factor=1, total_iters=self.args['lr_scheduler_warmup_epochs'])
        scheduler_anneal = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=self.args['lr_scheduler_end_factor'], total_iters=self.args['lr_scheduler_anneal_epochs'])
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler_warmup, scheduler_anneal], milestones=[self.args['lr_scheduler_warmup_epochs']])
        return [optimizer], [scheduler]
    
    def on_before_optimizer_step(self, optimizer):
        # Track the gradient norms
        grad_norms = grad_norm(self, norm_type=2)['grad_2.0_norm_total']
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=1)
        self.log('train/grad_norms', grad_norms, batch_size=1)
    
    # Forward passes, losses and inference
    def forward(self, x):
        # x: [batch_size, 2]
        x = self.model(x)
        return x
    
    def batch_to_loss_and_log(self, batch, log_name):
        x, y = batch
        batch_size = x.shape[0]
        pred = self.forward(x)
        loss = torch.nn.functional.mse_loss(pred, y)

        if log_name != None:
            self.log(f'{log_name}/mse_loss', loss, batch_size=batch_size) # Automatically averaged across all samples in batch. Averaged across all batches for val/test.
            self.log(f'{log_name}/monitor', loss, batch_size=batch_size) # Keep the best checkpoint based on this metric
        return loss, batch, pred

    # Step functions
    def training_step(self, batch, batch_idx):
        loss, _, _ = self.batch_to_loss_and_log(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.batch_to_loss_and_log(batch, 'val')
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, _, _ = self.batch_to_loss_and_log(batch, 'test')
        return loss
    
    def predict_step(self, batch, batch_idx):
        loss, batch, pred = self.batch_to_loss_and_log(batch, None)
        
        # Save the predictions/GT
        for i in range(pred.shape[0]):
            x, y = batch
            x = x[i].tolist()
            y = y[i].item()
            y_hat = pred[i].item()
            with open(f'{self.output_folder}/{batch_idx}_{i}.txt', 'a') as f:
                f.write(f'x: {x}\ny: {y}\ny_hat: {y_hat}\n\n')
        return 