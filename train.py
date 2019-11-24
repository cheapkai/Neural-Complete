import math
from typing import List

import numpy as np
import torch
import torch.nn

import parser.load
from lab.experiment.pytorch import Experiment
from model import SimpleLstmModel
from parser import tokenizer

# Configure the experiment

EXPERIMENT = Experiment(name="simple_lstm",
                        python_file=__file__,
                        comment="Simple LSTM",
                        check_repo_dirty=False,
                        is_log_python_file=False)

logger = EXPERIMENT.logger

# device to train on
device = torch.device("cuda:1")


def list_to_batches(x, batch_size, batches, seq_len):
    """
    Prepare flat data into batches to be ready for the model to consume
    """
    x = np.reshape(x, (batch_size, batches, seq_len))
    x = np.transpose(x, (1, 2, 0))

    return x


def get_batches(files: List[parser.load.EncodedFile], eof: int, batch_size=32, seq_len=32):
    """
    Covert raw encoded files into trainin/validation batches
    """

    # Shuffle the order of files
    np.random.shuffle(files)

    # Concatenate all the files whilst adding `eof` marker at the beginnings
    data = []
    for f in files:
        data.append(eof)
        data += f.codes
    data = np.array(data)

    # Start from a random offset
    offset = np.random.randint(seq_len * batch_size)
    data = data[offset:]

    # Number of batches
    batches = (len(data) - 1) // batch_size // seq_len

    # Extract input
    x = data[:(batch_size * seq_len * batches)]
    # Extract output, i.e. the next char
    y = data[1:(batch_size * seq_len * batches) + 1]

    # Covert the flat data into batches
    x = list_to_batches(x, batch_size, batches, seq_len)
    y = list_to_batches(y, batch_size, batches, seq_len)

    return x, y


class Trainer:
    """
    This will maintain states, data and train/validate the model
    """

    def __init__(self, *, files: List[parser.load.EncodedFile],
                 model, loss_func, optimizer,
                 eof: int,
                 batch_size: int, seq_len: int,
                 is_train: bool,
                 h0, c0):
        # Get batches
        x, y = get_batches(files, eof,
                           batch_size=batch_size,
                           seq_len=seq_len)
        # Covert data to PyTorch tensors
        self.x = torch.tensor(x, device=device)
        self.y = torch.tensor(y, device=device)

        # Initial state
        self.hn = h0
        self.cn = c0

        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.p = None
        self.is_train = is_train

    def run(self, i):
        # Get model output
        self.p, logits, (self.hn, self.cn) = self.model(self.x[i], self.hn, self.cn)

        # Flatten outputs
        logits = logits.view(-1, self.p.shape[-1])
        yi = self.y[i].view(-1)

        # Calculate loss
        loss = self.loss_func(logits, yi)

        # Store the states
        self.hn = self.hn.detach()
        self.cn = self.cn.detach()

        if self.is_train:
            # Take a training step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            logger.store("train_loss", loss.cpu().data.item())
        else:
            logger.store("valid_loss", loss.cpu().data.item())


def main_train():
    lstm_size = 1024
    lstm_layers = 3
    batch_size = 32
    seq_len = 32

    with logger.section("Loading data"):
        # Load all python files
        files = parser.load.load_files()
        # Split training and validation data
        train_files, valid_files = parser.load.split_train_valid(files, is_shuffle=False)

    with logger.section("Create model"):
        # Create model
        model = SimpleLstmModel(encoding_size=tokenizer.VOCAB_SIZE,
                                embedding_size=tokenizer.VOCAB_SIZE,
                                lstm_size=lstm_size,
                                lstm_layers=lstm_layers)
        # Move model to `device`
        model.to(device)

        # Create loss function and optimizer
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

    # Initial state is 0
    h0 = torch.zeros((lstm_layers, batch_size, lstm_size), device=device)
    c0 = torch.zeros((lstm_layers, batch_size, lstm_size), device=device)

    # Setup logger indicators
    logger.add_indicator("train_loss", queue_limit=500, is_histogram=True)
    logger.add_indicator("valid_loss", queue_limit=500, is_histogram=True)

    # Specify the model in [lab](https://github.com/vpj/lab) for saving and loading
    EXPERIMENT.add_models({'base': model})

    # Start training scratch (step '0')
    EXPERIMENT.start_train(0)

    # Number of batches per epoch
    batches = math.ceil(sum([len(f[1]) + 1 for f in train_files]) / (batch_size * seq_len))

    # Number of steps per epoch. We train and validate on each step.
    steps_per_epoch = 200

    # Train for 100 epochs
    for epoch in logger.loop(range(100)):
        # Create trainer
        trainer = Trainer(files=train_files,
                          model=model,
                          loss_func=loss_func,
                          optimizer=optimizer,
                          batch_size=batch_size,
                          seq_len=seq_len,
                          is_train=True,
                          h0=h0,
                          c0=c0,
                          eof=0)
        # Create validator
        validator = Trainer(files=valid_files,
                            model=model,
                            loss_func=loss_func,
                            optimizer=optimizer,
                            is_train=False,
                            seq_len=seq_len,
                            batch_size=batch_size,
                            h0=h0,
                            c0=c0,
                            eof=0)

        # Next batch to train and validation
        train_batch = 0
        valid_batch = 0

        # Loop through steps
        for i in range(1, steps_per_epoch):
            # Set global step
            global_step = epoch * batches + min(batches, (batches * i) // steps_per_epoch)
            logger.set_global_step(global_step)

            # Last batch to train and validate
            train_batch_limit = trainer.x.shape[0] * min(1., (i + 1) / steps_per_epoch)
            valid_batch_limit = validator.x.shape[0] * min(1., (i + 1) / steps_per_epoch)

            try:
                with logger.delayed_keyboard_interrupt():

                    with logger.section("train", total_steps=trainer.x.shape[0], is_partial=True):
                        model.train()
                        # Train
                        while train_batch < train_batch_limit:
                            trainer.run(train_batch)
                            logger.progress(train_batch + 1)
                            train_batch += 1

                    with logger.section("valid", total_steps=validator.x.shape[0], is_partial=True):
                        model.eval()
                        # Validate
                        while valid_batch < valid_batch_limit:
                            validator.run(valid_batch)
                            logger.progress(valid_batch + 1)
                            valid_batch += 1

                    # Output results
                    logger.write()

                    # 10 lines of logs per epoch
                    if (i + 1) % (steps_per_epoch // 10) == 0:
                        logger.new_line()

            except KeyboardInterrupt:
                logger.save_progress()
                logger.save_checkpoint()
                logger.new_line()
                return


if __name__ == '__main__':
    main_train()
