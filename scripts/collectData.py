import os
import argparse
import cv2
import numpy as np
import nibabel as nib
from glob import glob
from multiprocessing.dummy import Pool

parser = argparse.ArgumentParser(description='Collect Tumor/Organ Data')
parser.add_argument('--path', type=str, default='/data/kits19/data')

args = parser.parse_args()

def main():
    paths = glob(os.path.join(args.path, "*_*.npz"))
    
    organ_results = []
    tumor_results = []
    
    for path in paths:
        filename = os.path.basename(path)
        npz = np.load(path, allow_pickle=True)
        mask = npz.get('mask')

        if np.any(mask > 0):
            organ_results.append(filename)
        if np.any(mask > 1):
            tumor_results.append(filename)
    
    print("organ slices : ",len(organ_results))
    print("tumor slices :", len(tumor_results))
    np.save(os.path.join(paths, '%s_slices.npy' % ("organ")), organ_results)
    np.save(os.path.join(paths, '%s_slices.npy' % ("tumor")), tumor_results)

if __name__ == '__main__':
    main()
        