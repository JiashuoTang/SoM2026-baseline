import torch as th
import numpy as np
import argparse
import sys
import types
from torch.utils.data import Dataset

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def register_nyudepthv2_checkpoint_stub():
    nyu_module = types.ModuleType('nyudepthv2')
    def h5_loader(path):
        return None

    class NYUDataset(Dataset):
        def train_transform(self, *args):
            return None

        def val_transform(self, *args):
            return None

    nyu_module.h5_loader = h5_loader
    nyu_module.NYUDataset = NYUDataset
    sys.modules['nyudepthv2'] = nyu_module
