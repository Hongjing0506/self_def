'''
Author: ChenHJ
Date: 2021-12-12 21:40:15
LastEditors: ChenHJ
LastEditTime: 2022-03-13 23:00:03
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
def geo_ticks(axs, lonticks, latticks, cl, lonmin, latmin, extents):
    import proplot as pplt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter
    from cartopy.mpl.ticker import LatitudeFormatter
    from matplotlib.ticker import MultipleLocator
    
    axs.format(coast=True, coastlinewidth=0.8, coastzorder=1, coastcolor="grey6")
    proj = pplt.PlateCarree(central_longitude=cl)
    lonticks += -cl
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
            labelsize=7,
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
        
        
def lsmask(ds, lsdir, label):
    with xr.open_dataset(lsdir) as f:
        da = f["mask"][0]
    landsea = filplonlat(da)
    ds.coords["mask"] = (("lat", "lon"), landsea.values)
    if label == "land":
        ds = ds.where(ds.mask < 1)
    elif label == "ocean":
        ds = ds.where(ds.mask > 0)
    del ds["mask"]
    return ds


def filplonlat(ds):
    # To facilitate data subsetting
    # print(da.attrs)
    """
    print(
        f'\n\nBefore flip, lon range is [{ds["lon"].min().data}, {ds["lon"].max().data}].'
    )
    ds["lon"] = ((ds["lon"] + 180) % 360) - 180
    # Sort lons, so that subset operations end up being simpler.
    ds = ds.sortby("lon")
    """
    ds = ds.sortby("lat", ascending=True)
    # print(ds.attrs)
    print('\n\nAfter sorting lat values, ds["lat"] is:')
    print(ds["lat"])
    return ds

def plt_sig(da, ax, n, area, color, large):
    
    da_cyc, lon_cyc = add_cyclic_point(da[::n, ::n], coord=da.lon[::n])
    nx, ny = np.meshgrid(lon_cyc, da.lat[::n])
    sig = ax.scatter(
        nx[area], ny[area], marker=".", s=large, c=color, alpha=0.6)

def leadlag_array(da, n):
    new = da
    time = da.coords['time']
    if n > 0:
        # lag
        new[:-n] = np.array(da[n:])
        new[-n:] = np.nan
    if n < 0:
        # lead
        new[-n:] = np.array(da[:n])
        new[:-n] = np.nan
    new.coords['time'] = time
    return new
    
def add_shape(shp_path, ax, proj):
    from cartopy.io.shapereader import Reader
    reader = Reader(shp_path)
    shape = cfeature.ShapelyFeature(reader.geometries(), proj, edgecolor='grey6', facecolor='grey6')
    ax.add_feature(shape, linewidth=0.8)
    
def patches(x0, y0, width, height, proj):
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x0, y0),
        width,
        height,
        fc="none",
        ec="grey5",
        linewidth=1.0,
        zorder=1.1,
        transform=proj,
        linestyle="--",
    )
    ax.add_patch(rect)