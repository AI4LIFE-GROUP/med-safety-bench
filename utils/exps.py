import random
import numpy as np
import torch

import socket
import platform

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #safe to call this function even if cuda is not available



def log_info(logger, args, time_start):

    #show information about current job
    logger.info(f"UTC time (start): {time_start:%Y-%m-%d %H:%M:%S}")
    logger.info(f"Host: {socket.gethostname()}")
    
    # print all argparse'd args
    logger.info("\n------------------------------")
    logger.info("      Argparse arguments")
    logger.info("------------------------------")

    for arg in vars(args):
        logger.info(f"{arg} \t {getattr(args, arg)}")
    logger.info("------------------------------\n")

    # show information about device
    logger.info(f"torch.__version__ = {torch.__version__}\n")
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    if cuda_available:
        current_device = torch.cuda.current_device()
        current_device_properties = torch.cuda.get_device_properties(current_device)
        logger.info(f"Running on GPU: {current_device_properties}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"torch.version.cuda = {torch.version.cuda}")
        logger.info(f"Available CPU: {platform.processor()}")
    
    if not cuda_available:
        logger.info(f"Running on CPU: {platform.processor()}")
