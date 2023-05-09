import pickle
import sys
import json
import os
import random
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import torch
from path import Path
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

_PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()

sys.path.append(_PROJECT_DIR)
import glob
import numpy as np
import pandas as pd
import wandb
import math
from torch.nn.functional import normalize

import matplotlib
from matplotlib import pyplot as plt

random.seed(1)
np.random.seed(1)
num_clients = 10
num_classes = 10
dir_path = "cifar10/"

# def generate_figure(file_name, data):

    # print(glob.glob(_PROJECT_DIR+'/out/{}/{}/*'.format(data)))
    # for inx, name in glob.glob(_PROJECT_DIR+'/out/{}'):
# for label, acc in self.metrics.items():
#     if len(acc) > 0:
#         plt.plot(acc, label=label, ls=linestyle[label])
# plt.title(f"{self.algo} {self.args.dataset} acc")
# plt.ylim(0, 100)
# plt.xlabel("Communication Rounds")
# plt.ylabel("Accuracy")
# plt.grid()
# plt.legend()
# plt.savefig(
#     OUT_DIR / self.args.dataset / self.algo / f"gr{self.args.global_epoch}_le{self.args.local_epoch}_{self.args.ab}_{self.fn}_acc.jpeg", bbox_inches="tight"
# )
# plt.clf()
            
            # gr50_le20_B_lr01_acc_metrics
if __name__ == "__main__":
    file_name = sys.argv[1] if sys.argv[1] != None else False
    data = sys.argv[2] if sys.argv[2] != None else "cifar10"
    output_name = sys.argv[3] if sys.argv[3] != None else "test"
    gr = file_name.split("_")[0]
    le = file_name.split("_")[1]
    ab = file_name.split("_")[2]

    # generate_figure(file_name, data)
    # matplotlib.use("Agg")
    linestyle = {  
        "test_after": "solid",
        "train_after": "dotted",
        "val_after": "dashed",                              
    }
    print(_PROJECT_DIR)
    print(glob.glob(_PROJECT_DIR+'/out/cifar10/*'))
    dirs = glob.glob(_PROJECT_DIR+'/out/cifar10/*')

    for idx, file in enumerate(dirs):
        dir = file+'/'+file_name
        algo = file.split("/")[-1]
        file_data = pd.read_csv(dir)
        file_data.replace(0, np.nan, inplace=True)
        x=file_data['test_after']
        plt.plot( x, marker='.', label=algo)
        # plt.fill_between(np.linspace(1, rng, num=int(gr)-1), x.iloc[0] - x.iloc[1] ,  x.iloc[0] + x.iloc[1], alpha=0.2)


    plt.title(f"{gr}, {le} test acc")
    plt.ylim(0, 100)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig(
        # _PROJECT_DIR / self.args.dataset / self.algo / f"gr{self.args.global_epoch}_le{self.args.local_epoch}_{self.args.ab}_{self.fn}_acc.jpeg", bbox_inches="tight"
        _PROJECT_DIR / output_name+ "acc.jpeg", bbox_inches="tight"
    )
    plt.clf()
