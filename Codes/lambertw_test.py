# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:12:55 2022

@author: leisir
"""

from scipy.special import lambertw
import numpy as np

z = np.linspace(0,5,10)
z_lambert = np.real(lambertw(z))