'''
Author: ChenHJ
Date: 2021-12-12 21:40:15
LastEditors: ChenHJ
LastEditTime: 2021-12-17 13:22:06
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
note: xpos和ypos与生成heatmap的坐标轴刻度有关
param {axs object} ax 图
param {array} r 需要做检验的r
param {array} rlim 检验标准
param {float} xpos 调整*号的x轴位置
param {float} ypos 调整*号的y轴位置
param {str} size 调整*号的大小
return {*}
'''
def sign_check_star(ax, r, rlim, xpos, ypos, size):
    x = np.array(r)
    y = np.array(rlim)
    i_pass, j_pass = np.where(np.abs(x) >= y)
    for i,j in zip(i_pass,j_pass):
        ax.text(j+xpos, i+ypos, "*", size=size)
    del(x, y, i_pass, j_pass, i, j)
    
    
'''
description: 为输入的图增加lonticks和latticks并设置显示的区域
param {axs object} axs 图
param {array} lonticks 经度刻度
param {array} latticks 纬度刻度
param {int} cl 地图投影中心的经度
param {int} lonmin 经度次刻度间隔
param {int} latmin 纬度次刻度间隔
return {*}
'''    
def geo_ticks(axs, lonticks, latticks, cl, lonmin, latmin):
    import proplot as pplt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter
    from cartopy.mpl.ticker import LatitudeFormatter
    from matplotlib.ticker import MultipleLocator
    
    axs.format(coast=True, coastlinewidth=0.8, coastzorder=1)
    proj = pplt.PlateCarree(central_longitude=cl)
    lonticks += -cl
    print(lonticks)
    extents = [lonticks[0], lonticks[-1], latticks[0], latticks[-1]]
    axs.set_extent(extents, crs=proj)
    axs.set_xticks(lonticks, crs=proj)
    axs.set_yticks(latticks, crs=proj)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    axs.minorticks_on()
    xminorLocator = MultipleLocator(lonmin)
    yminorLocator = MultipleLocator(latmin)
    for ax in axs:
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
        ax.outline_patch.set_linewidth(1.0)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=8,
            direction="out",
            length=4.0,
            width=0.8,
            pad=2.0,
            top=False,
            right=False,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            length=3.0,
            width=0.8,
            top=False,
            right=False,
        )