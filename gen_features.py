import json
import pickle

import numpy as np
import torch
from tqdm import tqdm

from config import device, pickle_file
from utils import get_image

if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    print('loading model: {}...'.format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    with open('stars.json', encoding='utf-8') as file:
        stars = json.load(file)

    for star in tqdm(stars):
        name = star['name']
        file_list = star['file_list']
        feature_list = []
        for f in file_list:
            img = get_image(f)
            imgs = torch.zeros([1, 3, 112, 112], dtype=torch.float)
            imgs[0] = img
            with torch.no_grad():
                output = model(imgs)
                feature = output[0].cpu().numpy()
                x = feature / np.linalg.norm(feature)
                feature_list.append(x)
        star['feature_list'] = feature_list

    with open(pickle_file, 'wb') as file:
        pickle.dump(stars, file)
