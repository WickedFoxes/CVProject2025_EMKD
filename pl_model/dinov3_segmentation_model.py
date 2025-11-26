import torch
import torch.nn.functional as F

from models import get_model
from pl_model.base import BasePLModel
from datasets.dataset import SliceDataset, load_case_mapping, split_train_val

from torch.utils.data import DataLoader
from utils.loss_functions import calc_loss

class Dinov3SegmentationPLModel(BasePLModel):
    def __init__(self, params, train_indices, val_indices):
        super(Dinov3SegmentationPLModel, self).__init__()
        self.save_hyperparameters(params)
        self.net = get_model(self.hparams.model, channels=2, checkpoint_path=self.hparams.checkpoint_path)
        
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
        return DataLoader(
            dataset, 
            collate_fn=self.pad_collate_fn,
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True, 
            shuffle=True
        )

    def test_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.data_path,
            indices=self.val_indices,
            task=self.hparams.task,
            dataset=self.hparams.dataset,
            train=False
        )
        return DataLoader(
            dataset, 
            collate_fn=self.pad_collate_fn,
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True
        )

    def val_dataloader(self):
        return self.test_dataloader()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), 
            weight_decay=5e-2,    # 문구의 5 x 10^-2 반영
            betas=(0.9, 0.98)     # 문구의 설정 유지
        )
        scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=1e-6),
                     'interval': 'epoch',
                     'frequency': 1}
        return [opt], [scheduler]
    
    def pad_collate_fn(self, batch):
        # batch는 Dataset의 __getitem__이 리턴하는 항목들의 리스트입니다.

        batch_ct = []
        batch_mask = []
        batch_case = []
        
        patch_size = 28

        for ct, mask, case in batch:
            # 현재 이미지의 높이(h), 너비(w) 구하기
            # shape가 (C, H, W) 혹은 (H, W)라고 가정
            h, w = ct.shape[-2], ct.shape[-1]

            # 28로 나누어 떨어지기 위해 필요한 패딩 계산
            # (나머지가 0이면 패딩은 0이 됩니다)
            pad_h = (patch_size - (h % patch_size)) % patch_size
            pad_w = (patch_size - (w % patch_size)) % patch_size

            # 2. 패딩이 필요한 경우 양옆/위아래로 분배
            if pad_h > 0 or pad_w > 0:
                # 너비 패딩 분배
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left  # 홀수일 경우 오른쪽에 1픽셀 더 추가
                
                # 높이 패딩 분배
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top  # 홀수일 경우 아래쪽에 1픽셀 더 추가

                # 3. F.pad 적용 (순서: Left, Right, Top, Bottom)
                ct = F.pad(ct, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
                mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

            batch_ct.append(ct)
            batch_mask.append(mask)
            batch_case.append(case)

        batch_ct = torch.stack(batch_ct)      # (B, C, H, W)
        batch_mask = torch.stack(batch_mask)  # (B, C, H, W) 혹은 (B, H, W)

        return batch_ct, batch_mask, batch_case
