import logging
import os
import time

def create_log_dir(path, filename):
    if not os.path.exists(path):
        os.makedirs(path)

    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path + filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger



# log_path = './results/debug/logger/'
# logger = create_log_dir(log_path)
#
#
# for epoch in range(10):
#     loss = epoch * 10
#     mse = 1 + epoch
#
#     log = 'epoch' + str(epoch) + ':\t' + 'loss' + str(loss)
#
#     logger.info(log)