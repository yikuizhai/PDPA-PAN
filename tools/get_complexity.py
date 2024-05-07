import argparse
import jittor as jt
jt.flags.use_cuda_managed_allocator = 1
from jdet.runner import Runner
from jdet.config import init_cfg

import numpy as np
import random
import os


def main():
    config_file = "/media/msi/linux/project/JDet/configs/ITS_ablation/finish/test/oriented-test.py"

    jt.flags.use_cuda = 1

    init_cfg(config_file)

    runner = Runner()


    runner.get_para()



if __name__ == "__main__":
    main()
