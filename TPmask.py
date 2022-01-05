'''
Author: ChenHJ
Date: 2022-01-05 17:21:06
LastEditors: ChenHJ
LastEditTime: 2022-01-05 17:45:02
FilePath: /chenhj/self_def/TPmask.py
Aim: 
Mission: 
'''
# %%
import numpy as np
import xarray as xr
import os
import re
from cdo import Cdo
import shutil
import sys

sys.path.append("/home/ys17-23/chenhj/self_def/")
import plot as sepl
import cal as ca
import pandas as pd
from importlib import reload

# sd.path.append("/home/ys17-23/chenhj/1201code/self_def.py")

cdo = Cdo()

# for plot
import proplot as pplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter
from cartopy.mpl.ticker import LatitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from scipy import stats

reload(ca)
# %%
fhgt = xr.open_dataset("/home/ys17-23/chenhj/self_def/hgt.sfc.nc")
hgt = fhgt['hgt'][0,:,:].drop('time')
mask = 1013.25*(1.0-hgt*0.0065/288.15)**5.25145




# %%
