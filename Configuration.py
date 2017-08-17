import logging
import time

from Utils import Utils

class Configuration:
    def __init__(self):
        pass

    HIGGS_ALL = 'HIGGS.csv.gz'
    # HIGGS_ALL = 'HIGGS_10000.csv'
    # HIGGS_ALL = 'sim.csv'
    DATA_DIR = './data/'
    LOGGER_NAME = 'higgs_main_logger'
    LOG_DIR = './logs/'
    RESULTS_DIR = './results2/'
    HIGGS_FRACS = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    HIGGS_FRACS_TEST = [0.005]

    REGENERATE_DATA = False
	
    HYPEROPT_EVALS_PER_SEARCH = 25
    ANN_OPIMIZER_MAX_ITERATIONS = 50
    DNN_EPOCHS = 30

    ALL_DATA_FRACTION = 0.25 # how much data to use from entire higgs dataset (11M records)
    TEST_DATA_FRACTION = 0.25 # test data fraction out of entire dataset
    VALID_DATA_FRACTION = 0.25 # valid data fraction out of test dataset

    # which columns should be taken to process
    # from 0 to last column, higgs data has 28 features so columns 1:27
    # higgs dataset : 0 column - label, 1-21 coulmns - 21 low level features, 22-28 columns - 7 high level features
    FEATURES_START_COL = 22 #  including
    FEATURES_END_COL = 28 #  including 

    #  KEYS
    SVM_KEY = 'svm'
    ANN_KEY = 'ann'
    FOREST_KEY = 'random_forest'
    TREE_KEY = 'decision_tree'
    DNN_KEY = 'dnn'

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


def logger():
    return logging.getLogger(Configuration.LOGGER_NAME)

