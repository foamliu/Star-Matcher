import json
import os
from shutil import copyfile

from config import IMG_FOLDER

if __name__ == "__main__":
    stars = []
    dir_list = [d for d in os.listdir(IMG_FOLDER) if
                os.path.isdir(os.path.join(IMG_FOLDER, d))]
    for i, d in enumerate(dir_list):
        dir = os.path.join(IMG_FOLDER, d)
        new_dir = dir.replace(d, str(i))
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

        file_list = [os.path.join(dir, f) for f in os.listdir(dir) if f.lower().endswith('.jpg')]
        new_list = []
        for src in file_list:
            dst = src.replace(d + '_', '').replace(d, str(i)).replace('\\', '/')
            new_list.append(dst)
            copyfile(src, dst)
        stars.append({'name': d, 'file_list': new_list})

    # print(stars)
    with open('stars.json', 'w', encoding='utf-8') as file:
        json.dump(stars, file, ensure_ascii=False, indent=4)
