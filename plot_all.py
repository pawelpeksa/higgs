# -*- coding: utf-8 -*-

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
matplotlib.rc('font', family='Arial')
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
    l = 11 * len(higgs_fracs)
    fig = plt.figure(figsize=(15, l))

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

    plt.savefig(plot_dir + 'all.png')
    plt.clf()

    logger().info('plotting seperately finished')

    auc_dict_high = dict()
    auc_dict_low = dict()

    create_arrays(auc_dict_high)
    create_arrays(auc_dict_low)

    fig = plt.figure(figsize=(8, 6))

    for higgs_frac in higgs_fracs:
        logger().info('Plotting into summary for frac:%f', higgs_frac)

        results_high = open_results('./results_high/resultDict_' + str(higgs_frac) + '.dat')
        results_low = open_results('./results_low/resultDict_' + str(higgs_frac) + '.dat')  

        auc_dict_high[Config.TREE_KEY].append(auc_roc(results_high[Config.TREE_KEY]))
        auc_dict_high[Config.FOREST_KEY].append(auc_roc(results_high[Config.FOREST_KEY]))
        auc_dict_high[Config.ANN_KEY].append(auc_roc(results_high[Config.ANN_KEY]))
        auc_dict_high[Config.DNN_KEY].append(auc_roc(results_high[Config.DNN_KEY]))

        auc_dict_low[Config.TREE_KEY].append(auc_roc(results_low[Config.TREE_KEY]))
        auc_dict_low[Config.FOREST_KEY].append(auc_roc(results_low[Config.FOREST_KEY]))
        auc_dict_low[Config.ANN_KEY].append(auc_roc(results_low[Config.ANN_KEY]))
        auc_dict_low[Config.DNN_KEY].append(auc_roc(results_low[Config.DNN_KEY]))

    plot_all_summary(auc_dict_low, auc_dict_high, higgs_fracs)
    plt.savefig(plot_dir + 'summary.pdf')


def plot_all_summary(auc_dict_low, auc_dict_high, fracs):  
    plt.title(u'Zależność AUC ROC od ilości wykorzystanych danych')
    plt.xlabel(u'a')
    plt.ylabel(u'AUC ROC')

    lw = 1
    
    plt.plot(fracs, auc_dict_low[Config.TREE_KEY], lw=lw, color='b', label=Config.TREE_KEY + ' LL')   
    plt.plot(fracs, auc_dict_low[Config.FOREST_KEY], lw=lw, color='g', label=Config.FOREST_KEY + ' LL')   
    plt.plot(fracs, auc_dict_low[Config.ANN_KEY], lw=lw, color='c', label=Config.ANN_KEY + ' LL')   
    plt.plot(fracs, auc_dict_low[Config.DNN_KEY], lw=lw, color='r', label=Config.DNN_KEY + ' LL')   

    linestyle = '--'

    plt.plot(fracs, auc_dict_high[Config.TREE_KEY], lw=lw, linestyle=linestyle, color='b', label=Config.TREE_KEY + ' HL')   
    plt.plot(fracs, auc_dict_high[Config.FOREST_KEY], lw=lw, linestyle=linestyle, color='g', label=Config.FOREST_KEY + ' HL')   
    plt.plot(fracs, auc_dict_high[Config.ANN_KEY], lw=lw, linestyle=linestyle, color='c', label=Config.ANN_KEY + ' HL')   
    plt.plot(fracs, auc_dict_high[Config.DNN_KEY], lw=lw, linestyle=linestyle, color='r', label=Config.DNN_KEY + ' HL')   

    plt.legend(loc="lower right", ncol=2)

def auc_roc(psys):
    ps, ys = psys
    fpr, tpr, _ = roc_curve(ys, ps)
    return auc(fpr, tpr)


def create_arrays(dict):
    dict[Config.TREE_KEY] = list()
    dict[Config.FOREST_KEY] = list()
    dict[Config.ANN_KEY] = list()
    dict[Config.DNN_KEY] = list()


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
    plt.xlabel(u'False Positive Rate')
    plt.ylabel(u'True Positive Rate')

    if is_high:
        plt.title(u'Krzywa ROC dla a=%g przy użyciu cech wysokiego poziomu' % frac)
    else:
        plt.title(u'Krzywa ROC dla a=%g przy użyciu cech niskiego poziomu' % frac)

    plt.legend(loc="lower right")


if __name__ == '__main__':
    main()

