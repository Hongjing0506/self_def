"""
Author: ChenHJ
Date: 2021-12-03 11:45:38
LastEditors: ChenHJ
LastEditTime: 2021-12-03 14:54:28
FilePath: /chenhj/self_def/self_def.py
Aim: 
Mission: 
"""
# %%
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

"""
description: 
    (已废弃，请使用p_month)
    该函数用于选取数据中每一年的相应月份，并可选地计算每一年这些相应月份的平均值；例如：p_time(data, 6, 8, False)将选取数据中每一年的JJA，但不计算每年JJA的平均值；p_time(data, 6, 8, True)则在选取数据中每一年的JJA后计算每年JJA的平均值；
param {*} data
    the data should be xarray dataset or dataarray
param {float} mon_s
    the start months
param {float} mon_end
    the end months
param {bool} meanon
    whether to calculate the seasonal mean for every year
return {*}
    xarray dataarray
"""


def p_time(data, mon_s, mon_end, meanon):
    time = data["time"]
    n_data = data.sel(
        time=(data.time.dt.month <= mon_end) * (data.time.dt.month >= mon_s)
    )
    n_mon = mon_end - mon_s + 1
    if meanon == True:
        n_data_mean = n_data.coarsen(time=n_mon).mean()
        return n_data_mean
    elif meanon == False:
        return n_data
    else:
        print("Bad argument: meanon")


"""
description: 
    本函数用于将需要的月份挑选出来，并存储为月份x年份xlatxlon的形式
param {*} data
param {*} mon_s
param {*} mon_e
return {*}
"""


def p_month(data, mon_s, mon_e):
    import pandas as pd
    import xarray as xr

    time = data["time"]
    data.transpose("time", ...)
    year_s = pd.to_datetime(time).year[1]
    year_e = pd.to_datetime(time).year[-1]
    nyear = pd.date_range(str(year_s), str(year_e), freq="AS")
    m_ind = data.groupby("time.month").groups[mon_s]
    res = data[m_ind]
    res["time"] = nyear
    for i in np.arange(mon_s + 1, mon_e + 1):
        m_ind = data.groupby("time.month").groups[i]
        tmp = data[m_ind]
        tmp["time"] = nyear
        res = xr.concat([res, tmp], "month")

    month = np.arange(mon_s, mon_e + 1)
    res["month"] = month
    return res


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


def mapart(ax, extents):
    proj = ccrs.PlateCarree()
    ax.coastlines(color="k", lw=1.5)
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="white")
    ax.set_extent(extents, crs=proj)
    xticks = np.arange(extents[0], extents[1] + 1, 20)
    yticks = np.arange(extents[2], extents[3] + 1, 10)
    # 这里的间隔需要根据自己实际调整
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    xminorLocator = MultipleLocator(5)
    yminorLocator = MultipleLocator(10)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_minor_locator(xminorLocator)
    ax.xaxis.set_minor_locator(yminorLocator)
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
    # 为了便于在不同的场景中使用，这里使用了一个全局变量gl_font
    ax.minorticks_on()
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3.0,
        width=0.8,
        top=False,
        right=False,
    )
    ax.outline_patch.set_linewidth(1.0)


def detrend_dim(da, dim, deg, trend):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=1, skipna=True)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    if trend == False:
        return da - fit
    elif trend == True:
        return fit


def dim_linregress(x, y):
    # returns: slope,intercept,rvalue,pvalue,hypothesis
    return xr.apply_ufunc(
        stats.linregress,
        x,
        y,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask="parallelized",
    )


def plt_sig(da, ax, n, area):
    da_cyc, lon_cyc = add_cyclic_point(da[::n, ::n], coord=da.lon[::n])
    nx, ny = np.meshgrid(lon_cyc, da.lat[::n])
    sig = ax.scatter(
        nx[area], ny[area], marker=".", s=9, c="black", alpha=0.6, transform=proj
    )


def standardize(da):
    mean = da.mean(dim="time", skipna=True)
    std = da.std(dim="time", skipna=True)
    return (da - mean) / std


"""
description: 
    计算SAM index，v为monthly meridional wind，并需要包含850hPa和200hPa；本计算已进行纬度加权；
param {*} v
return {*}
"""


def SAM(v):
    # V850 - V200
    lon = v.lon
    lat = v.lat
    lon_range = lon[(lon >= 70.0) & (lon <= 110.0)]
    lat_range = lat[(lat >= 10.0) & (lat <= 30.0)]
    v850 = v.sel(level=850, lon=lon_range, lat=lat_range).drop("level")
    v200 = v.sel(level=200, lon=lon_range, lat=lat_range).drop("level")

    weights = np.cos(np.deg2rad(v850.lat))
    weights.name = "weights"
    v850_weighted_mean = v850.weighted(weights).mean(("lon", "lat"), skipna=True)
    v200_weighted_mean = v200.weighted(weights).mean(("lon", "lat"), skipna=True)
    # sam = standardize(rmmean(v850_weighted_mean) - rmmean(v200_weighted_mean))
    sam = v850 - v200
    del (
        lon,
        lat,
        lon_range,
        lat_range,
        v850,
        v200,
        weights,
        v850_weighted_mean,
        v200_weighted_mean,
    )
    return sam


"""
description: 
    计算SEAM index，u为monthly zonal wind，并需要包含850hPa层次；本计算已进行纬度加权；
param {*} u
return {*}
"""


def SEAM(u):
    lon = u.lon
    lat = u.lat
    lon_range1 = lon[(lon >= 90.0) & (lon <= 130.0)]
    lat_range1 = lat[(lat >= 5.0) & (lat <= 15.0)]
    lon_range2 = lon[(lon >= 110.0) & (lon <= 140.0)]
    lat_range2 = lat[(lat >= 22.5) & (lat <= 32.5)]
    u1 = u.sel(level=850, lon=lon_range1, lat=lat_range1).drop("level")
    u2 = u.sel(level=850, lon=lon_range2, lat=lat_range2).drop("level")

    weights1 = np.cos(np.deg2rad(u1.lat))
    weights1.name = "weights"
    weights2 = np.cos(np.deg2rad(u2.lat))
    weights2.name = "weights"

    u1_weighted_mean = u1.weighted(weights1).mean(("lon", "lat"), skipna=True)
    u2_weighted_mean = u2.weighted(weights2).mean(("lon", "lat"), skipna=True)

    # seam = standardize(rmmean(u1_weighted_mean) - rmmean(u2_weighted_mean))
    seam = u1 - u2
    del (
        lon,
        lat,
        lon_range1,
        lat_range1,
        lon_range2,
        lat_range2,
        u1,
        u2,
        weights1,
        weights2,
        u1_weighted_mean,
        u2_weighted_mean,
    )
    return seam


"""
description: 
param {*} u
return {*}
"""


def EAM(u):
    lon = u.lon
    lat = u.lat
    lon_range2 = lon[(lon >= 110.0) & (lon <= 150.0)]
    lat_range2 = lat[(lat >= 25.0) & (lat <= 35.0)]
    lon_range1 = lon[(lon >= 110.0) & (lon <= 150.0)]
    lat_range1 = lat[(lat >= 40.0) & (lat <= 50.0)]
    u1 = u.sel(level=200, lon=lon_range1, lat=lat_range1).drop("level")
    u2 = u.sel(level=200, lon=lon_range2, lat=lat_range2).drop("level")

    weights1 = np.cos(np.deg2rad(u1.lat))
    weights1.name = "weights"
    weights2 = np.cos(np.deg2rad(u2.lat))
    weights2.name = "weights"

    u1_weighted_mean = u1.weighted(weights1).mean(("lon", "lat"), skipna=True)
    u2_weighted_mean = u2.weighted(weights2).mean(("lon", "lat"), skipna=True)

    # eam = standardize(rmmean(u1_weighted_mean) - rmmean(u2_weighted_mean))
    eam = u1 - u2

    del (
        lon,
        lat,
        lon_range1,
        lat_range1,
        lon_range2,
        lat_range2,
        u1,
        u2,
        weights1,
        weights2,
        u1_weighted_mean,
        u2_weighted_mean,
    )
    return eam


"""
description: 
    计算Webster-Yang index；u为zonal wind，需包含850hPa和200hPa
param {*} u
return {*}
"""


def WY(u):
    lon = u.lon
    lat = u.lat
    lon_range1 = lon[(lon >= 40.0) & (lon <= 110.0)]
    lat_range1 = lat[(lat >= 5.0) & (lat <= 20.0)]

    u850 = u.sel(level=850, lon=lon_range1, lat=lat_range1).drop("level")
    u200 = u.sel(level=200, lon=lon_range1, lat=lat_range1).drop("level")

    weights1 = np.cos(np.deg2rad(u850.lat))
    weights1.name = "weights"
    weights2 = np.cos(np.deg2rad(u200.lat))
    weights2.name = "weights"

    u850_weighted_mean = u850.weighted(weights1).mean(("lon", "lat"), skipna=True)
    u200_weighted_mean = u200.weighted(weights2).mean(("lon", "lat"), skipna=True)

    # seam = standardize(rmmean(u850_weighted_mean) - rmmean(u200_weighted_mean))
    wyindex = u850_weighted_mean - u200_weighted_mean
    del (
        lon,
        lat,
        lon_range1,
        lat_range1,
        u850,
        u200,
        weights1,
        weights2,
        u850_weighted_mean,
        u200_weighted_mean,
    )
    return wyindex


"""
description: 用于将1D的月数据转换为季节数据，DJF已考虑过渡年，因此作为损失，最后一年将只保留1、2月用于计算倒数第二年的DJF；输入的数据，可以以2月份作为时间结尾，也可以以12月份作为数据结尾；
param {*} da
return {*}
"""


def mon_to_season1D(da):
    da.transpose("time", ...)
    time = da.coords["time"]
    nyear = pd.to_datetime(time).year[-1] - pd.to_datetime(time).year[1]
    year = pd.date_range(
        start=str(pd.to_datetime(time).year[1]),
        end=str(pd.to_datetime(time).year[-1] - 1),
        freq="AS",
    )
    try:
        if pd.to_datetime(time).month[-1] == 2:
            timesel = time[2:]
        elif pd.to_datetime(time).month[-1] == 12:
            timesel = time[2:-10]
        else:
            raise ValueError("the time of variable should be end in FEB or DEC")
    except ValueError as e:
        print("error: ", repr(e))
    da = da.sel(time=timesel).coarsen(time=3).mean()
    season = ["MAM", "JJA", "SON", "DJF"]
    temp = np.array(da)
    temp = np.reshape(temp, (4, nyear), order="F")
    nda = xr.DataArray(temp, coords=[season, year], dims=["season", "time"])
    del (time, nyear, year, timesel, season, temp)
    return nda


"""
description: 用于将3D的月数据转换为季节数据，DJF已考虑过渡年，因此作为损失，最后一年将只保留1、2月用于计算倒数第二年的DJF；输入的数据，可以以2月份作为时间结尾，也可以以12月份作为数据结尾；
param {*} da
return {*}
"""


def mon_to_season3D(da):
    da.transpose("time", "level", "lat", "lon")
    time = da.coords["time"]
    level = da.coords["level"]
    lat = da.coords["lat"]
    lon = da.coords["lon"]
    nyear = pd.to_datetime(time).year[-1] - pd.to_datetime(time).year[1]
    year = pd.date_range(
        start=str(pd.to_datetime(time).year[1]),
        end=str(pd.to_datetime(time).year[-1] - 1),
        freq="AS",
    )
    try:
        if pd.to_datetime(time).month[-1] == 2:
            timesel = time[2:]
        elif pd.to_datetime(time).month[-1] == 12:
            timesel = time[2:-10]
        else:
            raise ValueError("the time of variable should be end in FEB or DEC")
    except ValueError as e:
        print("error: ", repr(e))
    da = da.sel(time=timesel).coarsen(time=3).mean()
    season = ["MAM", "JJA", "SON", "DJF"]
    temp = np.array(da)
    temp = np.reshape(
        temp, (4, nyear, level.shape[0], lat.shape[0], lon.shape[0]), order="F"
    )
    nda = xr.DataArray(
        temp,
        coords=[season, year, level, lat, lon],
        dims=["season", "time", "level", "lat", "lon"],
    )
    del (time, level, lat, lon, nyear, year, timesel, season, temp)
    return nda


def leadlag_reg(x, y, freq, ll):
    if freq == "season":
        rvalue = np.zeros((4, 2 * ll + 1))
        pvalue = np.zeros((4, 2 * ll + 1))
        trvalue = np.zeros((4, 2 * ll + 1))


# %%

# print(np.reshape(x, (4, 42), order="F"))

# %%
# lat = [0.0, 1.0]
# lon = [0.0, 1.0]
# level = [850, 200]
# time = np.arange(1, 9, 1)

# x_n = np.reshape(x, (4, 2, 2, 2, 2), order="F")
# print(x[:, 0, 0, 0])
# print(x_n[1, :, 0, 0, 0])
# %%
