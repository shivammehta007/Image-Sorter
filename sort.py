import argparse
import logging
import os
from shutil import copyfile

import torch
from PIL import Image
from tqdm import tqdm
from train import Trainer

from config import LOGGING_LEVEL, LOGGING_FORMAT
from imagetransformers import testtransform
from model.ConvolutionalClassifier import ConvolutionalClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to Sort the folder')

    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="Path to the folder to sort the images"
    )

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Path to the trained pytorch model"
    )

    args = parser.parse_args()
    path = args.path

    model = ConvolutionalClassifier()
    trainer = Trainer(model, 0.001, 0.001)
    model = trainer.load_model(args.model).model

    filenames = []
    for (dirpath, dirnames, f) in os.walk(path):
        filenames.extend(f)

    needed_foldername = os.path.join(path, "Needed")
    not_needed_foldername = os.path.join(path, "Not Needed")

    if not os.path.exists(needed_foldername):
        os.mkdir(needed_foldername)
    if not os.path.exists(not_needed_foldername):
        os.mkdir(not_needed_foldername)

    for filename in tqdm(filenames):
        if filename == ".DS_Store":
            continue
        image = Image.open(os.path.join(path, filename))
        x = testtransform(image).unsqueeze(0)
        output = model(x)
        _, preds_tensor = torch.max(output, 1)

        final_class = preds_tensor[0].item()

        if final_class == 0:
            copyfile(os.path.join(path, filename), os.path.join(needed_foldername, filename))
        else:
            copyfile(os.path.join(path, filename), os.path.join(not_needed_foldername, filename))

    logger.info("Successfully Copied")
