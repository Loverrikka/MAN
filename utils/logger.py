import logging
import os
import sys
import os.path as osp
import datetime

# def setup_logger(name, save_dir, if_train):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)

#     ch = logging.StreamHandler(stream=sys.stdout)
#     ch.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     if save_dir:
#         if not osp.exists(save_dir):
#             os.makedirs(save_dir)
#         if if_train:
#             fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
#         else:
#             fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
#         fh.setLevel(logging.DEBUG)
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)

#     return logger

def setup_logger(name, save_dir, cfg, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get current date, hour, minute, and second
        current_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Construct log file name
        log_file_name = f"{current_date_time}_{'train' if if_train else 'test'}_log.txt"

        fh = logging.FileHandler(os.path.join(save_dir, log_file_name), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
