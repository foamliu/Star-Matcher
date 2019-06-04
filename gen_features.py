import json
import pickle

import numpy as np
import torch
from tqdm import tqdm

from config import device, pickle_file, star_file
from utils import get_image, FaceNotFoundError


def count_files(stars):
    num_files = 0
    for star in stars:
        num_files += len(star['file_list'])
    return num_files


if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    print('loading model: {}...'.format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    with open('stars.json', encoding='utf-8') as file:
        stars = json.load(file)

    num_files = count_files(stars)
    print('num_files: ' + str(num_files))

    print('Generating features...')
    for star in tqdm(stars):
        name = star['name']
        file_list = star['file_list']
        feature_list = []
        to_remove = []
        for f in file_list:
            try:
                img = get_image(f)
                imgs = torch.zeros([1, 3, 112, 112], dtype=torch.float)
                imgs[0] = img
                with torch.no_grad():
                    output = model(imgs)
                    feature = output[0].cpu().numpy()
                    x = feature / np.linalg.norm(feature)
                    feature_list.append(x)
            except FaceNotFoundError:
                to_remove.append(f)

        for f in to_remove:
            star['file_list'].remove(f)
        star['feature_list'] = feature_list

    with open(pickle_file, 'wb') as file:
        pickle.dump(stars, file)

    num_files = count_files(stars)
    print('num_files: ' + str(num_files))

    with open(pickle_file, 'rb') as file:
        stars = pickle.load(file)

    features = np.empty((num_files, 512), dtype=np.float32)
    files = []
    names = []

    i = 0
    print('Generating stars.pkl...')
    for star in stars:
        name = star['name']
        file_list = star['file_list']
        feature_list = star['feature_list']
        for idx, feature in enumerate(feature_list):
            features[i] = feature
            files.append(file_list[idx])
            names.append(name)
            i += 1

    print(features.shape)
    assert (len(names) == num_files)

    with open(star_file, 'wb') as file:
        save = dict()
        save['features'] = features
        save['files'] = files
        save['names'] = names
        pickle.dump(save, file)
