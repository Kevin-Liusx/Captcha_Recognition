import os
import glob
import numpy as np
import torch

from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing

import config 
import dataset

def run_training():
    # Get the image files, excluding the duplicate images
    train_image_files = [f for f in glob.glob(os.path.join(config.TRAIN_DATA_DIR, "*.png")) if "(1)" not in os.path.basename(f)]
    test_image_files = [f for f in glob.glob(os.path.join(config.TEST_DATA_DIR, "*.png")) if "(1)" not in os.path.basename(f)]

    # Get the targets, which are the names of the images without the .png extension
    train_targets_orig = [x.split("/")[-1][:-6] for x in train_image_files]
    test_targets_orig = [x.split("/")[-1][:-6] for x in test_image_files]
 
    # Convert the targets into a list of lists
    train_targets = [[c for c in x] for x in train_targets_orig]
    test_targets = [[c for c in x] for x in test_targets_orig]

    # Flatten the list of lists
    train_targets_flat = [c for clist in train_targets for c in clist]
    test_targets_flat = [c for clist in test_targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(train_targets_flat)
    
    # Encode the targets, add 1 to the result to make the padding value 0. 
    train_targets_enc = [lbl_enc.transform(x) + 1 for x in train_targets]
    test_targets_enc = [lbl_enc.transform(x) + 1 for x in test_targets]

    train_max_length = max(len(seq) for seq in train_targets_enc)

    # Add padding to the sequences, this is the final target.
    train_targets_enc_padded = np.array([np.pad(seq, (0, train_max_length - len(seq)), constant_values=-1) for seq in train_targets_enc])
    test_targets_enc_padded = np.array([np.pad(seq, (0, train_max_length - len(seq)), constant_values=-1) for seq in test_targets_enc])

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_image_files,
        targets=train_targets_enc_padded,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True
    )
    
    test_dataset = dataset.ClassificationDataset(
        image_paths=test_image_files,
        targets=test_targets_enc_padded,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False
    )
    
    # model = config.MODEL
    # model.to(config.DEVICE)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode="min", 
    #     patience=5, 
    #     factor=0.3, 
    #     verbose=True
    # )

    # for epoch in range(config.EPOCHS):
    #     model.train()
    #     for data in train_loader:
    #         images = data["image"]
    #         targets = data["targets"]

    #         images = images.to(config.DEVICE, dtype=torch.float)
    #         targets = targets.to(config.DEVICE, dtype=torch.long)

    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = loss_fn(outputs, targets)
    #         loss.backward()
    #         optimizer.step()

    #     model.eval()
    #     fin_targets = []
    #     fin_outputs = []
    #     with torch.no_grad():
    #         for data in test_loader:
    #             images = data


image_files = [f for f in glob.glob(os.path.join(config.TRAIN_DATA_DIR, "*.png")) if "(1)" not in os.path.basename(f)]
targets_orig = [x.split("/")[-1][:-6] for x in image_files]

targets = [[c for c in x] for x in targets_orig]
targets_flat = [c for clist in targets for c in clist]

lbl_enc = preprocessing.LabelEncoder()
lbl_enc.fit(targets_flat)
targets_enc = [lbl_enc.transform(x) + 1 for x in targets]
max_length = max(len(seq) for seq in targets_enc)
targets_enc_padded = np.array([np.pad(seq, (0, max_length - len(seq)), constant_values=-1) for seq in targets_enc])

print(len(lbl_enc.classes_))