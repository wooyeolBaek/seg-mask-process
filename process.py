import os
import glob
import heapq
import concurrent.futures
from collections import defaultdict, deque

import cv2
import numpy as np
from tqdm import tqdm



def get_regions(mask):

    assert len(mask.shape) == 2, AssertionError(f"mask.shape should be 2. not {len(mask.shape)}")
    height, width = mask.shape

    visited = np.zeros_like(mask, dtype=bool)
    result_dict = defaultdict(list)
    # dirs = [[-1,0],[0,-1],[0,1],[1,0]]
    dirs = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
    # dirs = [ # 방법 2: 거리 2만큼 탐색
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
                
                result_dict[target_value].append((np.array(xs), np.array(ys)))
    
    # sort ascending(avoid overlapped small region)
    for k,v in result_dict.items():
        result_dict[k].sort(key=lambda x:len(x[0]), reverse=True)    

    return result_dict


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

    result_dict = get_regions(mask)

    nmask = mask.copy()
    h, w = mask.shape
    area = h * w
    for pixel_value, regions in result_dict.items():
        for region in regions:
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





def main():
    base_path = "/Users/wooyeolbaek/Documents/Codes/seg-mask-process/masks"
    thr = .05
    thr = .1
    binary = False
    multiprocess = False
    multiprocess = True

    # get mask dirs
    mask_dirs = []
    for ext in ["png","jpg","jpeg"]:
        mask_dirs += glob.glob(base_path + f"/**.{ext}")

    base_name = os.path.basename(base_path)
    new_name = base_name + "_processed"
    save_dir = os.path.join(os.path.dirname(base_path), new_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if multiprocess:
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()//2) as executor:
            futures = []
            for mask_dir in mask_dirs:
                futures.append(executor.submit(process, mask_dir, save_dir, thr, binary))

            print("Processing futures ...")
            for future in tqdm(futures, total=len(futures)):
                future.result()
    else:
        for mask_dir in tqdm(mask_dirs,total=len(mask_dirs)):
            process(mask_dir, save_dir, thr, binary)
        


if __name__ == "__main__":
    main()