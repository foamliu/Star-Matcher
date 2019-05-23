import math
import pickle

import numpy as np
import torch

from config import device, pickle_file, num_files
from utils import get_image, get_prob

if __name__ == "__main__":
    with open(pickle_file, 'rb') as file:
        stars = pickle.load(file)

    features = np.empty((num_files, 512), dtype=np.float32)
    names = []

    i = 0
    for star in stars:
        name = star['name']
        feature_list = star['feature_list']
        for feature in feature_list:
            features[i] = feature
            names.append(name)
            i += 1

    print(features.shape)
    assert (len(names) == num_files)

    checkpoint = 'BEST_checkpoint.tar'
    print('loading model: {}...'.format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    test_fn = 'images/test_img_3.jpg'
    img = get_image(test_fn)
    imgs = torch.zeros([1, 3, 112, 112], dtype=torch.float)
    imgs[0] = img
    with torch.no_grad():
        output = model(imgs)
        feature = output[0].cpu().numpy()
        x = feature / np.linalg.norm(feature)

    cosine = np.dot(features, x)
    cosine = np.clip(cosine, -1, 1)
    print('cosine.shape: ' + str(cosine.shape))
    max_index = np.argmax(cosine)
    max_value = cosine[max_index]
    print('max_index: ' + str(max_index))
    print('max_value: ' + str(max_value))
    print(names[max_index])
    theta = math.acos(max_value)
    theta = theta * 180 / math.pi

    print('theta: ' + str(theta))
    prob = get_prob(theta)
    print('prob: ' + str(prob))
