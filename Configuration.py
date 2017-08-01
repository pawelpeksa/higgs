import logging
import time

from Utils import Utils

class Configuration:
    def __init__(self):
        pass

    HIGGS_ALL = 'HIGGS.csv.gz'
    # HIGGS_SMALL = 'HIGGS_10000.csv'
    DATA_DIR = './data/'
    LOGGER_NAME = 'higgs_main_logger'
    LOG_DIR = './logs/'

    REGENERATE_DATA = False

    ALL_DATA_FRACTION = 0.25 # how much data to use from entire higgs dataset (11M records)
    TEST_DATA_FRACTION = 0.25 # test data fraction out of entire dataset
    VALID_DATA_FRACTION = 0.25 # valid data fraction out of test dataset

    # which columns should be taken to process
    # from 0 to last column, higgs data has 28 features so columns 0:27
    # higgs dataset : 1 column - label, 2-22 coulmns - 21 low level features, 22-28 columns - high level features
    # TODO: bug here, start column now has to be always 0
    FEATURES_START_COL = 0 
    FEATURES_END_COL = 21

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

