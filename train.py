import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import LOGGING_LEVEL, LOGGING_FORMAT, TRAINED_MODEL_PATH, epochs, learning_rate, run_name, dropout, \
    regularisation
from model.ConvolutionalClassifier import ConvolutionalClassifier
from dataloaders import get_data_loaders
from imagetransformers import traintransform, testtransform

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


class Trainer:
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.losses = {'train': [], 'validation': []}

    def save_model(self, epoch, loss, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss}, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss

    def train(self, dataset):

        train_loss = 0
        self.model.train()
        for batch_idx, (data, target) in enumerate(tqdm(dataset)):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

        self.losses['train'].append(train_loss)

        return train_loss

    def test(self, dataset):
        valid_loss = 0.0
        self.model.eval()
        for batch_idx, (data, target) in enumerate(tqdm(dataset)):
            output = self.model(data)
            loss = self.criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

        self.losses['validation'].append(valid_loss)

        return valid_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Application to train a CNN based classifier')

    parser.add_argument(
        "-n",
        "--epochs",
        default=epochs,
        help="Number of Epochs to train the model for",
        type=int,
    )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=learning_rate,
        help="Learning rate of optimizer selected",
        type=float,
    )

    parser.add_argument(
        "-reg",
        "--regularisation",
        default=regularisation,
        help="L2 Weight decay values",
        type=float,
    )

    parser.add_argument(
        "-d",
        "--dropout",
        default=dropout,
        help="Dropout Value",
        type=float,
    )

    parser.add_argument(
        "-r",
        "--run-name",
        default=run_name,
        help="The name of the run for tensorboard logging",
    )

    args = parser.parse_args()
    logger.info("The Arguments are: {}".format(args))
    logger.info("Loading Dataset")

    logger.debug("Loading Transformation")
    train_transformation = traintransform
    test_transformation = testtransform

    train_loader, test_loader = get_data_loaders(train_transformation, test_transformation)



    logger.debug("Model Initializing")
    model = ConvolutionalClassifier(args.dropout)
    logger.info("Model Initialised")
    logger.debug(model)
    model_save_path = os.path.join(TRAINED_MODEL_PATH, args.run_name + ".pt")
    logger.info("Model will be saved at {}".format(model_save_path))

    trainer = Trainer(model, args.learning_rate, args.regularisation)

    test_loss_min = float("inf")
    for epoch in range(args.epochs):
        logger.info("\n========= Epoch %d of %d =========" % (epoch + 1, args.epochs))
        train_loss = trainer.train(train_loader)
        test_loss = trainer.test(test_loader)

        logger.info("\n========= Results: epoch %d of %d =========" % (epoch + 1, args.epochs))
        logger.info("train loss: %.2f" % train_loss)
        logger.info("test loss: %.2f" % test_loss)

        if test_loss <= test_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                test_loss_min,
                test_loss))
            trainer.save_model(epoch, train_loss, model_save_path)
            test_loss_min = test_loss