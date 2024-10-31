import torch
EPOCHS = 20
BATCH_SIZE = 16
TRAIN_DATA_DIR = '/Users/kevin/Desktop/CS4243/Captcha_Recognition/data/train'
TEST_DATA_DIR = '/Users/kevin/Desktop/CS4243/Captcha_Recognition/data/test'
IMAGE_WIDTH = 780
IMAGE_HEIGHT = 80
NUM_WORKERS = 8
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
