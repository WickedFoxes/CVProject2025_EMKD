import os
import argparse
import cv2
import numpy as np
import nibabel as nib
from glob import glob
from multiprocessing.dummy import Pool


parser = argparse.ArgumentParser(description='Slice KITS')
parser.add_argument('--in_path', type=str, default='/data/kits19/data')
parser.add_argument('--out_path', type=str, default='/data/kits')
parser.add_argument('--process_num', type=int, default=2)
parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()


def main():
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    out_path = os.path.join(args.out_path, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    paths = glob(os.path.join(args.in_path, "case_*/imaging*.nii.gz"))
    paths = [p for p in paths 
            if os.path.basename(os.path.dirname(p)) <= 'case_00209']

    # 0~167 (case_00167 이하)를 train_files 리스트로 생성
    if args.mode == 'train':
        paths = [p for p in paths 
                    if os.path.basename(os.path.dirname(p)) <= 'case_00167']
    elif args.mode == 'valid':
        # 168~209 (case_00167 초과)를 valid_files 리스트로 생성
        paths = [p for p in paths 
                    if os.path.basename(os.path.dirname(p)) > 'case_00167']

    pool = Pool(args.process_num)

    organ_result, tumor_result = pool.map(make_slice, paths)
    save_info(out_path, organ_result, task="organ")
    save_info(out_path, tumor_result, task="tumor")


def make_slice(path):
    """
    Cut 3D kits data into 2D slices
    :param path: /*/*.nii.gz
    :return: Slices and Infos
    """
    case, vol, seg = read_kits(path)

    organ_result = []
    tumor_result = []

    for i in range(vol.shape[0]):
        ct_slice = vol[i, ...]
        if ct_slice.shape != [512, 512]:
            ct_slice = cv2.resize(ct_slice, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        mask_slice = seg[i, ...]
        np.savez_compressed(f'{args.out_path}/{args.mode}/{case}_{i}.npz', ct=ct_slice, mask=mask_slice)
        if np.any(mask_slice > 0):
            organ_result.append(f'{case}_{i}.npz')
        if np.any(mask_slice > 1):
            tumor_result.append(f'{case}_{i}.npz')

    print(f'complete making KITS {args.mode} slices of {case}')
    return organ_result, tumor_result


def read_kits(path):
    dir = os.path.dirname(path)
    vol = nib.load(path).get_fdata()
    seg = nib.load(os.path.join(dir, 'segmentation.nii.gz')).get_fdata().astype('int8')
    case = os.path.split(dir)[-1][-5:]
    return case, vol, seg


def save_info(save_path, result, task:str="tumor"):
    slices = []
    for i in result:
        slices += i
    np.save(os.path.join(save_path, '%s_%s_slices.npy' % ("kits", task)), slices)


if __name__ == '__main__':
    main()