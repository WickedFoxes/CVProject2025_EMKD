import torch
import torch.nn.functional as F

from models import get_model
from pl_model.base import BasePLModel
from datasets.dataset import SliceDataset

from torch.utils.data import DataLoader
from utils.loss_functions import calc_loss

class Dinov3SegmentationPLModel(BasePLModel):
    def __init__(self, params, train_indices, val_indices):
        super(Dinov3SegmentationPLModel, self).__init__()
        self.save_hyperparameters(params)
        
        # ViT/DINO 계열은 체크포인트 로드가 중요하므로 해당 인자 유지
        self.net = get_model(
            self.hparams.model, 
            channels=2, 
            checkpoint_path=self.hparams.vit_checkpoint_path
        )
        
        self.train_indices = train_indices
        self.val_indices = val_indices

    def forward(self, x):
        # 모델이 (output, aux1, aux2) 형태의 튜플을 반환한다고 가정
        output = self.net(x)
        # output, _, _ = self.net(x)
        return output

    def training_step(self, batch, batch_idx):
        ct, mask, name = batch
        output = self.forward(ct)
        loss = calc_loss(output, mask)  # Dice_loss Used

        # [수정 1] Loss 로깅 추가 (Lightning 2.x 권장)
        # on_epoch=True: epoch 단위 평균 자동 계산
        # prog_bar=True: 진행바에 표시
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # [수정 2] 딕셔너리가 아닌 loss 텐서 반환
        return loss

    def validation_step(self, batch, batch_idx):
        # BasePLModel의 measure 메서드를 활용하기 위해 test_step 호출
        self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        ct, mask, name = batch
        output = self.forward(ct)

        # BasePLModel에 정의된 measure 메서드로 Dice Score 등 계산
        self.measure(batch, output)

    def train_dataloader(self):
        dataset = SliceDataset(
            data_path=self.hparams.data_path,
            indices=self.train_indices,
            task=self.hparams.task,
            dataset=self.hparams.dataset,
            train=True
        )
        # collate_fn이 필요한 경우 주석 해제하여 사용
        return DataLoader(
            dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True, 
            shuffle=True,
            # collate_fn=self.pad_collate_fn, 
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
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True,
            # collate_fn=self.pad_collate_fn,
        )

    def val_dataloader(self):
        return self.test_dataloader()

    def configure_optimizers(self):
        # [설정 유지] Transformer 계열 학습에 중요한 AdamW 설정 유지
        # betas=(0.9, 0.98)은 ViT 논문 등에서 자주 사용되는 설정입니다.
        opt = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=5e-2, 
            betas=(0.9, 0.98)  
        )
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.hparams.epochs, eta_min=1e-6
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        return [opt], [scheduler]