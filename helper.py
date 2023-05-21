#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import numpy as np

def F(l=0.235, w=0.15, r=0.0475):
    '''Compute F-matrix (3 controls * 4 wheels) of four mecanum wheel robot'''
    
    F = r/4 * np.array([[-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                        [       1,       1,       1,        1],
                        [      -1,       1,      -1,        1]])
    return F


def Tsc(x, y, theta):
    '''Calculate cube config SE(3) in {s}-frame given (x, y, theta)'''
    
    Tsc = np.array([[np.cos(theta), -np.sin(theta), 0,     x], 
                    [np.sin(theta),  np.cos(theta), 0,     y],
                    [            0,              0, 1, 0.025],
                    [            0,              0, 0,     1]])
    return Tsc


def writeConfiguration(configs, filename):
    '''Write all configs to csv file'''
    
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(configs)

