import os
import argparse
import cv2
import numpy as np
import nibabel as nib
from glob import glob
from multiprocessing.dummy import Pool


parser = argparse.ArgumentParser(description='Slice LITS')
parser.add_argument('--in_path', type=str, default='/data/lits/data')
parser.add_argument('--out_path', type=str, default='/data/lits')
parser.add_argument('--process_num', type=int, default=2)
parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()


def main():
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    out_path = os.path.join(args.out_path, args.mode)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if args.mode == 'train':
        paths = glob(os.path.join(args.in_path, "Training Batch 2", "volume-*.nii"))
    if args.mode == 'valid':
        paths = glob(os.path.join(args.in_path, "Training Batch 1", "volume-*.nii"))

    pool = Pool(args.process_num)

    organ_result, tumor_result = pool.map(make_slice, paths)
    save_info(paths, organ_result, task="organ")
    save_info(paths, tumor_result, task="tumor")


def make_slice(path):
    """
    Cut 3D lits data into 2D slices
    :param path: /*/*.nii.gz
    :return: Slices and Infos
    """
    case, vol, seg = read_lits(path)

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

    print(f'complete making slices of {case}')
    return result

def read_lits(path):
    vol = nib.load(path).get_fdata()
    seg = nib.load(path.replace('volume', 'segmentation')).get_fdata().astype('int8')
    case = path.split('-')[-1].split('.')[0]
    vol = np.transpose(vol, (2, 0, 1))
    seg = np.transpose(seg, (2, 0, 1))
    return case, vol, seg

def save_info(save_path, result, task:str="tumor"):
    slices = []
    for i in result:
        slices += i
    np.save(os.path.join(save_path, '%s_%s_slices.npy' % ("lits", task)), slices)


if __name__ == '__main__':
    main()