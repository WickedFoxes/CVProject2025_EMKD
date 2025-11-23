import torch
from models import get_model
from pl_model.base import BasePLModel
from datasets.dataset import SliceDataset, load_case_mapping, split_train_val

from torch.utils.data import DataLoader
from utils.loss_functions import calc_loss

class SegmentationPLModel(BasePLModel):
    def __init__(self, params, train_indices, val_indices):
        super(SegmentationPLModel, self).__init__()
        self.save_hyperparameters(params)
        self.net = get_model(self.hparams.model, channels=2)
        
        self.train_indices = train_indices
        self.val_indices = val_indices


    def forward(self, x):
        output, _, _ = self.net(x)
        return output

    def training_step(self, batch, batch_idx):
        ct, mask, name = batch
        output = self.forward(ct)
        loss = calc_loss(output, mask)  # Dice_loss Used

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        ct, mask, name = batch
        output = self.forward(ct)

        self.measure(batch, output)

    def train_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.data_path,
            indices=self.train_indices,
            task=self.hparams.task,
            dataset=self.hparams.dataset,
            train=True
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)

    def test_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.data_path,
            indices=self.val_indices,
            task=self.hparams.task,
            dataset=self.hparams.dataset,
            train=False
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return self.test_dataloader()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=1e-6),
                     'interval': 'epoch',
                     'frequency': 1}
        return [opt], [scheduler]