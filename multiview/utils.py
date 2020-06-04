import pickle
import os
import pathlib
import logging
import numpy as np

# logging.basicConfig(filename='logs/data_generation.log', level=logging.INFO,
#                     format='%(levelname)s:%(name)s:%(message)s')

strBold = lambda skk: "\033[1m {}\033[00m".format(skk)
strBlue = lambda skk: "\033[34m {}\033[00m".format(skk)
strRed = lambda skk: "\033[91m {}\033[00m".format(skk)
strGreen = lambda skk: "\033[92m {}\033[00m".format(skk)
strYellow = lambda skk: "\033[93m {}\033[00m".format(skk)
strLightPurple = lambda skk: "\033[94m {}\033[00m".format(skk)
strPurple = lambda skk: "\033[95m {}\033[00m".format(skk)
strCyan = lambda skk: "\033[96m {}\033[00m".format(skk)
strLightGray = lambda skk: "\033[97m {}\033[00m".format(skk)
strBlack = lambda skk: "\033[98m {}\033[00m".format(skk)

prBold = lambda skk: print("\033[1m {}\033[00m".format(skk))
prBlue = lambda skk: print("\033[34m {}\033[00m".format(skk))
prRed = lambda skk: print("\033[91m {}\033[00m".format(skk))
prGreen = lambda skk: print("\033[92m {}\033[00m".format(skk))
prYellow = lambda skk: print("\033[93m {}\033[00m".format(skk))
prLightPurple = lambda skk: print("\033[94m {}\033[00m".format(skk))
prPurple = lambda skk: print("\033[95m {}\033[00m".format(skk))
prCyan = lambda skk: print("\033[96m {}\033[00m".format(skk))
prLightGray = lambda skk: print("\033[97m {}\033[00m".format(skk))
prBlack = lambda skk: print("\033[98m {}\033[00m".format(skk))


def create_logger(log_file):
    """
    create logger for different purpose
    :param log_file: the place to store the log
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s-%(levelname)s-%(name)s]:%(message)s")

    file_handler = logging.FileHandler("{}.log".format(log_file))
    file_handler.setLevel(logging.INFO)  # only INFO in file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.ERROR)  # show ERROR on console
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def sigmoid(x, derivative=False):
    """
    compute the sigmoid function 1 / (1 + exp(-x))
    :param x: input of sigmoid function
    :param derivative: boolean value, if True compute the derivative of sigmoid function instead
    :return:
    """
    if x > 100:
        sigm = 1.
    elif x < -100:
        sigm = 0.
    else:
        sigm = 1. / (1. + np.exp(-x))

    if derivative:
        return sigm * (1. - sigm)
    return sigm
