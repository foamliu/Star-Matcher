import json
import os
from shutil import copyfile
from tqdm import tqdm
from config import IMG_FOLDER

if __name__ == "__main__":
    IMG_FOLDER_OLD = 'data/weiweiimage_old'

    stars = []
    dir_list = [d for d in os.listdir(IMG_FOLDER_OLD) if
                os.path.isdir(os.path.join(IMG_FOLDER_OLD, d))]

    num_files = 0

    for i in tqdm(range(len(dir_list))):
        d = dir_list[i]
        dir = os.path.join(IMG_FOLDER_OLD, d)
        new_dir = os.path.join(IMG_FOLDER, str(i))
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

        file_list = [f for f in os.listdir(dir) if f.lower().endswith('.jpg')]
        new_list = []
        for f in file_list:
            src = os.path.join(dir, f)
            dst = os.path.join(new_dir, f.replace(d + '_', '')).replace('\\', '/')
            new_list.append(dst)
            copyfile(src, dst)
            num_files += 1
        stars.append({'name': d, 'file_list': new_list})

    # print(stars)
    with open('stars.json', 'w', encoding='utf-8') as file:
        json.dump(stars, file, ensure_ascii=False, indent=4)

    print('num_stars: ' + str(len(stars)))
    print('num_files: ' + str(num_files))
