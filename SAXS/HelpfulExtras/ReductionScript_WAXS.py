#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:19:18 2025

@author: lauraforster
"""

import os
import numpy as np
from pathlib import Path
import h5py
from h5py import File
import matplotlib.pyplot as plt
import logging
# New path (pyFAI ≥ 2024.10)
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI import units
logging.getLogger("pyFAI").setLevel(logging.ERROR)
import warnings
from Utils import progress_bar
import time

        
def ReductionIQ(Filenumber, sample_loc, output_path, outputfolder,sample_beginning, sample_end, identifier, frame_index, nq, nchi, q_range, chi_range, calib, Plot,file_start=None, file_end=None, x_list=None, y_list=None):
    print('Beginning I_q integration')
    
    
    return 

