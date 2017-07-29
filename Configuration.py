import logging
import time

from Utils import Utils

class Configuration:
    def __init__(self):
        pass

    HIGGS_0_25 = 'data_025/HIGGS.csv'
    HIGGS_10000 = 'data_small/HIGGS.csv'
    HIGGS_ALL = 'data_all/HIGGS.csv.gz'
    DATA_DIR = './data/'
    LOGGER_NAME = 'higgs_main_logger'
    LOG_DIR = './logs/'

    @staticmethod
    def configure_logger():

        Configuration.maybe_create_log_dir()

        logger = logging.getLogger(Configuration.LOGGER_NAME)
        logger.setLevel(logging.DEBUG)

        log_filename = Configuration.generate_logfile_name()

        sh = logging.StreamHandler()
        fh = logging.FileHandler(Configuration.LOG_DIR + log_filename)

        sh.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')

        # add formatter to sh
        sh.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(sh)
        logger.addHandler(fh)

        logger.info("Logger configured")

    @staticmethod
    def maybe_create_log_dir():
        Utils.maybe_create_directory(Configuration.LOG_DIR)

    @staticmethod
    def generate_logfile_name():
        return time.strftime('%d_%m_%Y_%H_%M_%S' + '.log')

