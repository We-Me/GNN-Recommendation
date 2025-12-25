import os
import random
import numpy as np
import torch
import pytorch_lightning as pl


def seed_everything_strict(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    pl.seed_everything(seed, workers=True)
