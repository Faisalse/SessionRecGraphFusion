#!/usr/bin/env python36
# -*- coding: utf-8 -*-


"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time
from IGT import *
import pandas as pd

def main():
    train = pd.read_csv("diginetica_train_full.txt", sep="\t")
    test = pd.read_csv("diginetica_test.txt", sep="\t")
    
    igt = IGT(1,0.0001,1,64,0.00005,0.2)
    igt.fit(train, test)
    

if __name__ == '__main__':
    main()
