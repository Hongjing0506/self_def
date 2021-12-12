'''
Author: ChenHJ
Date: 2021-12-12 21:40:15
LastEditors: ChenHJ
LastEditTime: 2021-12-12 21:42:43
FilePath: /chenhj/self_def/plot.py
Aim: 
Mission: 
'''
import numpy as np
import xarray as xr
import os
import re
from cdo import Cdo
import shutil
import pandas as pd

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
from scipy.stats import t


'''
description: 对于heatmap，输入r和rlim进行显著性检验并对通过显著性检验的值加*号
param {*} ax 图
param {*} r 需要做检验的r
param {*} rlim 检验标准
param {*} xpos 调整*号的x轴位置
param {*} ypos 调整*号的y轴位置
param {*} size 调整*号的大小
return {*}
'''
def sign_check_star(ax, r, rlim, xpos, ypos, size):
    x = np.array(r)
    y = np.array(rlim)
    i_pass, j_pass = np.where(np.abs(x) >= y)
    for i,j in zip(i_pass,j_pass):
        print(i,j)
        ax.text(j+xpos, i+ypos, "*", size=size)