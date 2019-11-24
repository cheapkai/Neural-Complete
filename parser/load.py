from pathlib import Path
from typing import NamedTuple, List

import numpy as np

from lab.logger import Logger
from parser import tokenizer

logger = Logger()


class EncodedFile(NamedTuple):
    path: str
    codes: List[int]


def load_files() -> List[EncodedFile]:
    """
    Load encoded files
    """
    with open(str(Path(__file__).parent.parent / 'data' / 'all.py')) as f:
        lines = f.readlines()

    files = []
    for i in range(0, len(lines), 2):
        path = lines[i][:-1]
        content = lines[i + 1][:-1]
        if content == '':
            content = []
        else:
            content = [int(t) for t in content.split(' ')]
        files.append(EncodedFile(path, content))

    return files


def split_train_valid(files: List[EncodedFile], is_shuffle=True) -> (List[EncodedFile], List[EncodedFile]):
    """
    Split training and validation sets
    """
    if is_shuffle:
        np.random.shuffle(files)

    total_size = sum([len(f.codes) for f in files])
    valid = []
    valid_size = 0
    while len(files) > 0:
        if valid_size > total_size * 0.15:
            break
        valid.append(files[0])
        valid_size += len(files[0].codes)
        files.pop(0)

    train_size = sum(len(f.codes) for f in files)
    if train_size < total_size * 0.60:
        raise RuntimeError("Validation set too large")

    logger.info(train_size=train_size,
                valid_size=valid_size,
                vocab=tokenizer.VOCAB_SIZE)
    return files, valid


def main():
    files, code_to_str = load_files()
    logger.info(code_to_str)


if __name__ == "__main__":
    main()
