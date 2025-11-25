import os
import argparse

from pl_model.segmentation_model import SegmentationPLModel
from datasets.dataset import load_case_mapping, split_train_val

from sklearn.model_selection import KFold
import numpy as np

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser('train')
parser.add_argument('--data_path', type=str, default='/data/kits/data')
parser.add_argument('--checkpoint_path', type=str, default='/data/checkpoints')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--model', type=str, default='raunet')
parser.add_argument('--task', type=str, default='tumor', choices=['tumor', 'organ'])
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default='kits', choices=['kits', 'lits'])
parser.add_argument('--kfold', action='store_true', help='Enable 5-fold cross validation')

def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    
    case_mapping = load_case_mapping(args.data_path, args.task)
    train_indices, val_indices = split_train_val(
        case_mapping, train_ratio=0.8, seed=args.seed
    )

    model = SegmentationPLModel(args, train_indices, val_indices)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_path),
        filename='checkpoint_%s_%s_%s_{epoch}' % (args.dataset, args.task, args.model),
        save_last=True,
        save_top_k=5,
        monitor='dice_class0',
        mode='max',
        verbose=True
    )

    logger = TensorBoardLogger('log', name='%s_%s_%s' % (args.dataset, args.task, args.model))
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=args.epochs, 
        callbacks=[checkpoint_callback], 
        enable_progress_bar=False,
        logger=logger
    )
    trainer.fit(model)

def main_k_fold():
    args = parser.parse_args()
    seed_everything(args.seed)
    
    all_cases = load_case_mapping(args.data_path, args.task)
    
    case_ids = np.array(sorted(all_cases.keys()))
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(case_ids)):
        print(f"\nStart Training Fold: {fold} / 4")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        # 인덱스를 이용해 실제 데이터 ID 리스트 추출
        train_cases = case_ids[train_idx]
        val_cases = case_ids[val_idx]

        train_indices = []
        for case_id in train_cases:
            train_indices.extend(all_cases[case_id]['indices'])
            
        val_indices = []
        for case_id in val_cases:
            val_indices.extend(all_cases[case_id]['indices'])
        
        print(f" - Cases: Train {len(train_cases)}, Val {len(val_cases)}")
        print(f" - Slices: Train {len(train_indices)}, Val {len(val_indices)}")
        
        # 모델 초기화 (현재 Fold의 인덱스 전달)
        model = SegmentationPLModel(args, train_indices=train_indices, val_indices=val_indices)

        # Checkpoint: 파일명에 fold 정보를 포함시킵니다.
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_path, f'fold{fold}'), # 폴더를 fold별로 구분하거나
            filename=f'checkpoint_{args.dataset}_{args.task}_{args.model}_fold{fold}_' + '{epoch}', # 파일명에 fold 명시
            save_last=True,
            save_top_k=5,
            monitor='dice_class0',
            mode='max',
            verbose=True
        )

        # Logger: 버전 이름을 fold로 설정하여 텐서보드에서 겹치지 않게 합니다.
        logger = TensorBoardLogger(
            'log', 
            name=f'{args.dataset}_{args.task}_{args.model}',
            version=f'fold_{fold}' 
        )

        trainer = Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            max_epochs=args.epochs, 
            callbacks=[checkpoint_callback], 
            enable_progress_bar=False,
            logger=logger
        )
        
        # 학습 시작
        trainer.fit(model)
        
        # (선택 사항) 메모리 정리를 위해 모델과 트레이너 삭제 및 캐시 비우기
        del model, trainer
        torch.cuda.empty_cache()



def test():
    args = parser.parse_args()
    model = SegmentationPLModel.load_from_checkpoint(
        checkpoint_path=os.path.join(args.checkpoint_path, 'last.ckpt')
    )
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
    )
    trainer.test(model)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        if args.kfold:
            main_k_fold()
        else:
            main()
    if args.mode == 'test':
        test()