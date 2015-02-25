#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import print_function


import argparse
import sys
import numpy as np
import scipy
import pandas as pd
import librosa


def mk_tempo(infile=None, outfile=None, min_tempo=45.0):

    df = pd.read_table(infile, header=None, sep='\s+')

    # Build a KDE for tempo
    kernel = scipy.stats.gaussian_kde(df[0].diff().dropna())
    x = np.linspace(0, 60.0/min_tempo, num=1000)
    y = kernel(x)

    # Find the peaks of the density estimator
    idx = np.flatnonzero(librosa.util.localmax(y))

    # Candidate tempi
    x_cand = 60.0/x[idx]

    # And their densities
    y_cand = y[idx]

    # Find the top two candidates
    i = np.argsort(y_cand)[:-3:-1]

    x_best = x_cand[i]
    y_best = y_cand[i] / y_cand[i].sum()

    if len(i) == 1:
        x_best = np.pad(x_best, [(0, 1)], mode='constant')

    output = np.concatenate((x_best, [y_best[0]]))
    np.savetxt(outfile, output[np.newaxis, :], fmt='%.3f')


def get_params(args):

    parser = argparse.ArgumentParser(description='beat to tempo converter')

    parser.add_argument('infile', type=str,
                        help='Path to the input txt file')
    parser.add_argument('outfile', type=str,
                        help='Path to the output txt file')
    parser.add_argument('-t', '--min_tempo', type=float, default=45.0,
                        help='Minimum tempo')

    return vars(parser.parse_args(args))


if __name__ == '__main__':

    params = get_params(sys.argv[1:])

    mk_tempo(**params)
