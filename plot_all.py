import logging
import sys
import pickle
import math
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from Configuration import logger
from Configuration import Configuration as Config
from Utils import Utils

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') # has to be imported before pyplot
import matplotlib.pyplot as plt

results_high = './results_high/'
results_low = './results_low/'
plot_dir = './plots/'

def main():
    print 'plot all'

    Config.configure_logger()
    Utils.maybe_create_directory(plot_dir)
        
    higgs_fracs = Config.HIGGS_FRACS	

    logger().info('Plotting for fracs:')
    logger().info(higgs_fracs)

    plots_n = len(higgs_fracs)*2
    col_n = 2    	
    fig = plt.figure(figsize=(15, 11 * len(higgs_fracs)))

    i = 1

    for higgs_frac in higgs_fracs:
        results_high = open_results('./results_high/resultDict_' + str(higgs_frac) + '.dat')
        results_low = open_results('./results_low/resultDict_' + str(higgs_frac) + '.dat')

        
        plt.subplot(plots_n, col_n, i)
        i += 1

        plot_from_dict(results_low[Config.TREE_KEY], 'tree', higgs_frac, False)
        plot_from_dict(results_low[Config.FOREST_KEY], 'forest', higgs_frac, False)
        plot_from_dict(results_low[Config.ANN_KEY], 'ann', higgs_frac, False)
        plot_from_dict(results_low[Config.DNN_KEY], 'dnn', higgs_frac, False)

        plt.axis('equal')


        plt.subplot(plots_n, col_n, i)
        i += 1

        plot_from_dict(results_high[Config.TREE_KEY], 'tree', higgs_frac, True)
        plot_from_dict(results_high[Config.FOREST_KEY], 'forest', higgs_frac, True)
        plot_from_dict(results_high[Config.ANN_KEY], 'ann', higgs_frac, True)
        plot_from_dict(results_high[Config.DNN_KEY], 'dnn', higgs_frac, True)

        plt.axis('equal')

        
    fig.tight_layout()

    plt.savefig(plot_dir + 'all.pdf')

    logger().info('plotting finished finished')


def open_results(path):
    with open(path, 'r') as f:
        results = pickle.load(f)

    return results    


def plot_from_dict(psys, title, frac, is_high):
    ps, ys = psys
    plot_roc(ps, ys, title, frac, is_high)


def plot_roc(ps, ys, title, frac, is_high):
    logger().info('Plot:' + title)

    fpr, tpr, _ = roc_curve(ys, ps)
    roc_auc = auc(fpr, tpr)

    lw = 2

    label = title + ' ROC curve (area = %0.2f)' % roc_auc

    plt.plot(fpr, tpr, lw=lw, label=label)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if is_high:
        plt.title('Krzywa ROC dla a=%g przy uzyciu cech wysokiego poziomu' % frac)
    else:
        plt.title('Krzywa ROC dla a=%g przy uzyciu cech niskiego poziomu' % frac)

    plt.legend(loc="lower right")


if __name__ == '__main__':
    main()

