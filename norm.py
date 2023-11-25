import os
import cv2
import glob
import heapq
import concurrent
import numpy as np
from collections import defaultdict, deque
import concurrent.futures


def norm(image_dir, save_dir):
    mask_name = os.path.basename(image_dir)
    mask = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(save_dir, f"{os.path.splitext(mask_name)[0]}.png"),mask*255)


def main():
    base_path = "/Users/wooyeolbaek/Documents/Codes/seg-mask-process/masks_processed"

    # get mask dirs
    mask_dirs = []
    for ext in ["png","jpg","jpeg"]:
        mask_dirs += glob.glob(base_path + f"/**.{ext}")

    base_name = os.path.basename(base_path)
    new_name = base_name + "_norm"
    save_dir = os.path.join(os.path.dirname(base_path), new_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()//2) as executor:
        futures = []
        for mask_dir in mask_dirs:
            futures.append(executor.submit(norm, mask_dir, save_dir))

        for future in futures:
            future.result()
        

if __name__ == "__main__":
    main()