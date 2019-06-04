import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors
image_w = 112
image_h = 112
num_classes = 85164
num_files = 2165

DATA_DIR = 'data'
IMG_FOLDER = 'data/weiweiimage'

pickle_file = DATA_DIR + '/' + 'weiweiimage.pkl'
star_file = DATA_DIR + '/' + 'stars.pkl'
