import os
import glob
import heapq
import argparse
import concurrent.futures
from collections import defaultdict, deque

import cv2
import numpy as np
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser(description="inference")

    parser.add_argument(
        "--base_path",
        type=str,
        default="./masks",
    )
    parser.add_argument(
        "--thr",
        type=int,
        default=.1,
    )
    parser.add_argument(
        "--binary",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--multiprocess",
        type=bool,
        default=True,
    )

    args = parser.parse_args()

    return args


def get_regions(mask):

    assert len(mask.shape) == 2, AssertionError(f"mask.shape should be 2. not {len(mask.shape)}")
    height, width = mask.shape

    visited = np.zeros_like(mask, dtype=bool)
    result_list = []
    # dirs = [[-1,0],[0,-1],[0,1],[1,0]]
    dirs = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
    # dirs = [
    #     [-2,-2],[-2,-1],[-2,0],[-2,1],[-2,2],
    #     [-1,-2],[-1,-1],[-1,0],[-1,1],[-1,2],
    #     [0,-2],[0,-1],[0,1],[0,2],
    #     [1,-2],[1,-1],[1,0],[1,1],[1,2],
    #     [2,-2],[2,-1],[2,0],[2,1],[2,2],
    # ]

    # run bfs for each not visited pixel
    for h in range(height):
        for w in range(width):

            if not visited[h][w]:
                
                # bfs init
                q = deque([[h,w]])

                # save coords init
                target_value = mask[h][w]
                xs = [h]
                ys = [w]

                # bfs
                while q:
                    x, y = q.popleft()
                    for dx, dy in dirs:
                        nx = x + dx
                        ny = y + dy

                        if 0 <= nx < height and 0 <= ny < width and not visited[nx][ny] and mask[nx][ny] == target_value:
                            q.append([nx,ny])
                            visited[nx][ny] = True
                            xs.append(nx)
                            ys.append(ny)
                
                result_list.append([target_value, (np.array(xs), np.array(ys))])
    
    # sort ascending(avoid overlapped small region)
    result_list.sort(key=lambda x:len(x[1][0]), reverse=True)

    # return result_dict
    return result_list


def get_adj_pixels(region, mask):
    height, width = mask.shape

    # save # of neighbor pixels
    result_dict = defaultdict(int)

    # for checking whether the coords included in the region
    outside = np.ones_like(mask, dtype=bool)
    outside[region] = 0

    dirs = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]

    # # of neighbor pixels outside the region
    for x, y in zip(*region):
        
        for dx, dy in dirs:
            nx = x + dx
            ny = y + dy

            if 0<=nx<height and 0<=ny<width and outside[nx][ny]:
                result_dict[mask[nx][ny]] += 1
    
    # get the most frequent neighbor pixel
    hq = [(v,k) for k,v in result_dict.items()]
    heapq.heapify(hq)
    _, npixel_value = heapq.heappop(hq)

    return npixel_value


def process(image_dir, save_dir, thr=.1, binary=False):
    mask_name = os.path.basename(image_dir)
    mask = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)

    result_list = get_regions(mask)

    nmask = mask.copy()
    h, w = mask.shape
    area = h * w

    for pixel_value, region in result_list:

        if len(region[0]) > area * thr:
            continue

        npixel_value = get_adj_pixels(region, mask)
        npixel_value = get_adj_pixels(region, nmask)

        nmask[region] = npixel_value

    if binary:
        nmask[nmask>0]=255
        cv2.imwrite(os.path.join(save_dir, f"{os.path.splitext(mask_name)[0]}.png"),nmask)
    else:
        cv2.imwrite(os.path.join(save_dir, f"{os.path.splitext(mask_name)[0]}.png"),cv2.normalize(nmask, None, 0, 255, cv2.NORM_MINMAX))


def main(args):

    # get mask dirs
    mask_dirs = []
    for ext in ["png","jpg","jpeg"]:
        mask_dirs += glob.glob(args.base_path + f"/**.{ext}")

    base_name = os.path.basename(args.base_path)
    new_name = base_name + "_processed"
    save_dir = os.path.join(os.path.dirname(args.base_path), new_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.multiprocess:
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()//2) as executor:
            futures = []
            for mask_dir in mask_dirs:
                futures.append(executor.submit(process, mask_dir, save_dir, args.thr, args.binary))

            print("Processing futures ...")
            for future in tqdm(futures, total=len(futures)):
                future.result()
    else:
        for mask_dir in tqdm(mask_dirs,total=len(mask_dirs)):
            process(mask_dir, save_dir, args.thr, args.binary)
        


if __name__ == "__main__":
    args = parse_args()
    main(args)