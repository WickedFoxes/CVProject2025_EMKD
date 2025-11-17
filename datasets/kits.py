import os
import copy
import random
import numpy as np
import torch
import utils as du
from torch.utils.data import Dataset
# 'du' 모듈은 원본 코드에 있었던 의존성으로,
# window_standardize 및 cut_384 함수를 포함하고 있다고 가정합니다.
# import data_utils as du 

class KitsSliceDataset(Dataset):
    def __init__(self, data_path, task='tumor', series_index=None, train=True):
        """ 'kits' 데이터셋 전용 클래스 """
        super(KitsSliceDataset, self).__init__()
        assert task in ['organ', 'tumor']
        
        self.load_path = data_path
        self.series_index = series_index
        self.task = task
        self.train = train
        
        # 'kits' 슬라이스 목록 로드
        self.slice_list_path = os.path.join(data_path, 'kits_%s_slices.npy' % task)
        self.tumor_slices = np.load(self.slice_list_path)

    def rotate(self, img, mask, k=None):
        """ 90도 단위 회전 증강 """
        if k is None:
            k = random.choice([0, 1, 2, 3])
        img = np.rot90(img, k, (-2, -1))
        mask = np.rot90(mask, k, (-2, -1))
        return img, mask

    def flip(self, img, mask, flip=None):
        """ 좌우/상하 반전 증강 """
        if flip is None:
            a, b = random.choice([1, -1]), random.choice([1, -1])
        else:
            a, b = flip
        
        if img.ndim == 2:
            img = img[::a, ::b]
        elif img.ndim == 3: # (C, H, W)를 가정
            img = img[:, ::a, ::b]
        mask = mask[::a, ::b] # 마스크는 (C, H, W) 또는 (H, W)
        
        return img, mask

    def __len__(self):
        return len(self.tumor_slices)

    def __getitem__(self, item):
        # 1. 데이터 로딩
        f_name = self.tumor_slices[item]
        case = f_name.split('_')[0]
        npz_path = os.path.join(self.load_path, f_name)
        
        try:
            npz = np.load(npz_path, allow_pickle=True)
            ct = npz.get('ct')
            mask = npz.get('mask')
        except Exception as e:
            print(f"Error loading file: {npz_path}")
            raise e

        # 2. 태스크별 마스크 전처리
        if self.task == 'organ':
            mask[mask > 0] = 1
        elif self.task == 'tumor':
            mask = mask >> 1
            mask[mask > 0] = 1

        # 3. 데이터셋별 윈도잉 ('kits' 전용)
        # 'du' 모듈이 필요합니다.
        ct = du.window_standardize(ct, -200, 300) # <-- 'kits' 값 고정

        # 4. 데이터 증강 (훈련 시에만)
        if self.train:
            ct, mask = self.flip(ct, mask)
            ct, mask = self.rotate(ct, mask)

        # 5. 마스크 one-hot 인코딩 (배경/전경)
        img0 = copy.deepcopy(mask)
        img0 += 1
        img0[img0 != 1] = 0
        
        mask = np.stack((img0, mask), axis=0)
        mask[mask > 0] = 1

        # 6. 텐서 변환 및 크롭
        # 'du' 모듈이 필요합니다.
        ct = torch.from_numpy(du.cut_384(ct.copy())).unsqueeze(0).float()
        mask = torch.from_numpy(du.cut_384(mask.copy())).float()

        return ct, mask, case