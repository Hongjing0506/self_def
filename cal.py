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
from scipy.stats import t

"""
description: 
    (已废弃，请使用p_month)
    该函数用于选取数据中每一年的相应月份，并可选地计算每一年这些相应月份的平均值；例如: p_time(data, 6, 8, False)将选取数据中每一年的JJA，但不计算每年JJA的平均值；p_time(data, 6, 8, True)则在选取数据中每一年的JJA后计算每年JJA的平均值；
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
param {integer} mon_s
param {integer} mon_e
return {*}
"""


def p_month(data, mon_s, mon_e):
    import pandas as pd
    import xarray as xr

    time = data["time"]
    data.transpose("time", ...)
    year_s = pd.to_datetime(time).year[0]
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
    sam = v850_weighted_mean - v200_weighted_mean
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
    seam = u1_weighted_mean - u2_weighted_mean
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
    eam = u1_weighted_mean - u2_weighted_mean

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
    nyear = pd.to_datetime(time).year[-1] - pd.to_datetime(time).year[0]
    year = pd.date_range(
        start=str(pd.to_datetime(time).year[0]),
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
description: 用于将3D(time x lat x lon)的月数据转换为季节数据，DJF已考虑过渡年，因此作为损失，最后一年将只保留1、2月用于计算倒数第二年的DJF；输入的数据，可以以2月份作为时间结尾，也可以以12月份作为数据结尾；
param {*} da
return {*}
"""


def mon_to_season3D(da):
    da.transpose("time", "lat", "lon")
    time = da.coords["time"]
    lat = da.coords["lat"]
    lon = da.coords["lon"]
    nyear = pd.to_datetime(time).year[-1] - pd.to_datetime(time).year[0]
    year = pd.date_range(
        start=str(pd.to_datetime(time).year[0]),
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
    temp = np.reshape(temp, (4, nyear, lat.shape[0], lon.shape[0]), order="F")
    nda = xr.DataArray(
        temp, coords=[season, year, lat, lon], dims=["season", "time", "lat", "lon"],
    )
    del (time, lat, lon, nyear, year, timesel, season, temp)
    return nda


"""
description: 用于将4D(time x level x lat x lon)的月数据转换为季节数据，DJF已考虑过渡年，因此作为损失，最后一年将只保留1、2月用于计算倒数第二年的DJF；输入的数据，可以以2月份作为时间结尾，也可以以12月份作为数据结尾；
param {*} da
return {*}
"""


def mon_to_season4D(da):
    da.transpose("time", "level", "lat", "lon")
    time = da.coords["time"]
    level = da.coords["level"]
    lat = da.coords["lat"]
    lon = da.coords["lon"]
    nyear = pd.to_datetime(time).year[-1] - pd.to_datetime(time).year[0]
    year = pd.date_range(
        start=str(pd.to_datetime(time).year[0]),
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


"""
description: 用于计算超前滞后相关，同时将输出对应置信度的r临界值，其中x、y为已经处理为（季节x年份）的数据， freq="season"， ll表示超前（滞后）自身多少个时间单位， inan为在多余的表格中填充np.nan, clevel[0]为是否需要计算有效自由度, clevel[1]为置信度水平
param {*} x
param {*} y
param {str} freq: "season"
param {integer} ll
param {bool} inan
param {list} clevel
return {*}
"""


def leadlag_reg(x, y, freq, ll, inan, clevel):
    try:
        if freq == "season":
            x.transpose("season", "time")
            y.transpose("season", "time")
            x_season = x.coords["season"]
            y_season = y.coords["season"]
            x_nseason = x_season.shape[0]
            y_nseason = y_season.shape[0]
            nyear = x.coords["time"].shape[0]
            if ll != y_nseason:
                raise ValueError(
                    "ll should be less equal to the number of season in y}"
                )
            # elif ll == y_nseason:
            #     tmp = y_nseason
            # elif ll < y_nseason:
            #     tmp = ll
            avalue = np.zeros((x_nseason, 2 * ll + x_nseason), dtype=np.float64)
            bvalue = np.zeros((x_nseason, 2 * ll + x_nseason), dtype=np.float64)
            rvalue = np.zeros((x_nseason, 2 * ll + x_nseason), dtype=np.float64)
            pvalue = np.zeros((x_nseason, 2 * ll + x_nseason), dtype=np.float64)
            hyvalue = np.zeros((x_nseason, 2 * ll + x_nseason), dtype=np.float64)
            reff = np.zeros((x_nseason, 2 * ll + x_nseason), dtype=np.float64)
            tmp_time = np.arange(1, nyear, 1)
            for xsea in np.arange(0, x_nseason, 1):
                # calculate the lead-lag correlation of present year
                for ysea in np.arange(0, ll, 1):
                    (
                        avalue[xsea, ll + ysea],
                        bvalue[xsea, ll + ysea],
                        rvalue[xsea, ll + ysea],
                        pvalue[xsea, ll + ysea],
                        hyvalue[xsea, ll + ysea],
                    ) = dim_linregress(x[xsea, :], y[ysea, :])
                    if clevel[0] == True:
                        neff = eff_DOF(x[xsea, :], y[ysea, :], "1", 2)
                    elif clevel[0] == False:
                        neff = np.shape(x[xsea, :])[0]
                    t_lim = t.ppf(0.5 + 0.5 * clevel[1], neff)
                    reff[xsea, ll + ysea] = cal_rlim(t_lim, neff)

                    # calculate the lead correlation of last year
                    x_tmp = x[:, 1:]
                    y_tmp = y[:, :-1]
                    x_tmp.coords["time"], y_tmp.coords["time"] = tmp_time, tmp_time
                    (
                        avalue[xsea, ysea],
                        bvalue[xsea, ysea],
                        rvalue[xsea, ysea],
                        pvalue[xsea, ysea],
                        hyvalue[xsea, ysea],
                    ) = dim_linregress(x_tmp[xsea, :], y_tmp[ysea, :])
                    if clevel[0] == True:
                        neff = eff_DOF(x_tmp[xsea, :], y_tmp[ysea, :], "1", 2)
                    elif clevel[0] == False:
                        neff = np.shape(x_tmp[xsea, :])[0]
                    t_lim = t.ppf(0.5 + 0.5 * clevel[1], neff)
                    reff[xsea, ysea] = cal_rlim(t_lim, neff)

                    # calculate the lag correlation of next year
                    x_tmp = x[:, :-1]
                    y_tmp = y[:, 1:]
                    x_tmp.coords["time"], y_tmp.coords["time"] = tmp_time, tmp_time
                    (
                        avalue[xsea, 2 * ll + ysea],
                        bvalue[xsea, 2 * ll + ysea],
                        rvalue[xsea, 2 * ll + ysea],
                        pvalue[xsea, 2 * ll + ysea],
                        hyvalue[xsea, 2 * ll + ysea],
                    ) = dim_linregress(x_tmp[xsea, :], y_tmp[ysea, :])
                    if clevel[0] == True:
                        neff = eff_DOF(x_tmp[xsea, :], y_tmp[ysea, :], "1", 2)
                    elif clevel[0] == False:
                        neff = np.shape(x_tmp[xsea, :])[0]
                    t_lim = t.ppf(0.5 + 0.5 * clevel[1], neff)
                    reff[xsea, 2 * ll + ysea] = cal_rlim(t_lim, neff)

            if inan == True:
                for iiinan in np.arange(0, x_nseason - 1, 1):
                    (
                        avalue[iiinan, iiinan - ll + 1 :],
                        bvalue[iiinan, iiinan - ll + 1 :],
                        rvalue[iiinan, iiinan - ll + 1 :],
                        pvalue[iiinan, iiinan - ll + 1 :],
                        hyvalue[iiinan, iiinan - ll + 1 :],
                        reff[iiinan, iiinan - ll + 1 :],
                    ) = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

                    (
                        avalue[iiinan + 1, : iiinan + 1 :],
                        bvalue[iiinan + 1, : iiinan + 1 :],
                        rvalue[iiinan + 1, : iiinan + 1 :],
                        pvalue[iiinan + 1, : iiinan + 1 :],
                        hyvalue[iiinan + 1, : iiinan + 1 :],
                        reff[iiinan + 1, : iiinan + 1 :],
                    ) = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        else:
            raise ValueError(r"freq should be one of {season, year}")
    except ValueError as e:
        print(repr(e))
    del (
        x_season,
        y_season,
        x_nseason,
        y_nseason,
        nyear,
        xsea,
        ysea,
        x_tmp,
        y_tmp,
        tmp_time,
        neff,
    )
    return (avalue, bvalue, rvalue, pvalue, hyvalue, reff)


"""
description: 用于计算滑动超前滞后相关，其中x、y为已经处理为（季节x年份）的数据， freq="season"， ll表示超前（滞后）自身多少个时间单位， inan为在多余的表格中填充np.nan， window表示滑动窗口的长度
param {*} x
param {*} y
param {str} freq: "season"
param {integer} ll
param {bool} inan
param {integer} window
return {*}
"""


def leadlag_reg_rolling(x, y, freq, ll, inan, window):
    try:
        if freq == "season":
            x.transpose("season", "time")
            y.transpose("season", "time")
            x_season = x.coords["season"]
            y_season = y.coords["season"]
            x_nseason = x_season.shape[0]
            y_nseason = y_season.shape[0]
            nyear = x.coords["time"].shape[0]
            if ll != y_nseason:
                raise ValueError(
                    "ll should be less equal to the number of season in y}"
                )
            # elif ll == y_nseason:
            #     tmp = y_nseason
            # elif ll < y_nseason:
            #     tmp = ll
            avalue = np.zeros(
                (x_nseason, 2 * ll + x_nseason, nyear - (window - 1)), dtype=np.float64
            )
            bvalue = np.zeros(
                (x_nseason, 2 * ll + x_nseason, nyear - (window - 1)), dtype=np.float64
            )
            rvalue = np.zeros(
                (x_nseason, 2 * ll + x_nseason, nyear - (window - 1)), dtype=np.float64
            )
            pvalue = np.zeros(
                (x_nseason, 2 * ll + x_nseason, nyear - (window - 1)), dtype=np.float64
            )
            hyvalue = np.zeros(
                (x_nseason, 2 * ll + x_nseason, nyear - (window - 1)), dtype=np.float64
            )
            tmp_time = np.arange(1, nyear, 1)
            for xsea in np.arange(0, x_nseason, 1):
                # calculate the lead-lag correlation of present year
                for ysea in np.arange(0, ll, 1):
                    for nx in np.arange(0, nyear - window + 1, 1):
                        (
                            avalue[xsea, ll + ysea, nx],
                            bvalue[xsea, ll + ysea, nx],
                            rvalue[xsea, ll + ysea, nx],
                            pvalue[xsea, ll + ysea, nx],
                            hyvalue[xsea, ll + ysea, nx],
                        ) = dim_linregress(
                            x[xsea, nx : nx + window], y[ysea, nx : nx + window]
                        )

                        (
                            avalue[xsea, ysea, 0],
                            bvalue[xsea, ysea, 0],
                            rvalue[xsea, ysea, 0],
                            pvalue[xsea, ysea, 0],
                            hyvalue[xsea, ysea, 0],
                        ) = (np.nan, np.nan, np.nan, np.nan, np.nan)

                        (
                            avalue[xsea, 2 * ll + ysea, nx],
                            bvalue[xsea, 2 * ll + ysea, nx],
                            rvalue[xsea, 2 * ll + ysea, nx],
                            pvalue[xsea, 2 * ll + ysea, nx],
                            hyvalue[xsea, 2 * ll + ysea, nx],
                        ) = (np.nan, np.nan, np.nan, np.nan, np.nan)

                    for nx in np.arange(0, nyear - window, 1):
                        # calculate the lead correlation of last year
                        x_tmp = x[:, 1:]
                        y_tmp = y[:, :-1]
                        x_tmp.coords["time"], y_tmp.coords["time"] = tmp_time, tmp_time
                        (
                            avalue[xsea, ysea, nx + 1],
                            bvalue[xsea, ysea, nx + 1],
                            rvalue[xsea, ysea, nx + 1],
                            pvalue[xsea, ysea, nx + 1],
                            hyvalue[xsea, ysea, nx + 1],
                        ) = dim_linregress(
                            x_tmp[xsea, nx : nx + window], y_tmp[ysea, nx : nx + window]
                        )
                        # calculate the lag correlation of next year
                        x_tmp = x[:, :-1]
                        y_tmp = y[:, 1:]
                        x_tmp.coords["time"], y_tmp.coords["time"] = tmp_time, tmp_time
                        (
                            avalue[xsea, 2 * ll + ysea, nx],
                            bvalue[xsea, 2 * ll + ysea, nx],
                            rvalue[xsea, 2 * ll + ysea, nx],
                            pvalue[xsea, 2 * ll + ysea, nx],
                            hyvalue[xsea, 2 * ll + ysea, nx],
                        ) = dim_linregress(
                            x_tmp[xsea, nx : nx + window], y_tmp[ysea, nx : nx + window]
                        )

            if inan == True:
                for iiinan in np.arange(0, x_nseason - 1, 1):
                    (
                        avalue[iiinan, iiinan - ll + 1 :, :],
                        bvalue[iiinan, iiinan - ll + 1 :, :],
                        rvalue[iiinan, iiinan - ll + 1 :, :],
                        pvalue[iiinan, iiinan - ll + 1 :, :],
                        hyvalue[iiinan, iiinan - ll + 1 :, :],
                    ) = (np.nan, np.nan, np.nan, np.nan, np.nan)

                    (
                        avalue[iiinan + 1, : iiinan + 1 :, :],
                        bvalue[iiinan + 1, : iiinan + 1 :, :],
                        rvalue[iiinan + 1, : iiinan + 1 :, :],
                        pvalue[iiinan + 1, : iiinan + 1 :, :],
                        hyvalue[iiinan + 1, : iiinan + 1 :, :],
                    ) = (np.nan, np.nan, np.nan, np.nan, np.nan)

        else:
            raise ValueError(r"freq should be one of {season, year}")
    except ValueError as e:
        print(repr(e))
    del (
        x_season,
        y_season,
        x_nseason,
        y_nseason,
        nyear,
        xsea,
        ysea,
        x_tmp,
        y_tmp,
        tmp_time,
    )
    return (avalue, bvalue, rvalue, pvalue, hyvalue)


"""
description: 计算无偏估计的sigma^2
param {float/integer} n1: the number of array1
param {float/integer} n2: the number of array2
param {float} s1: standard deviation of array1
param {float} s2: standard deviation of array2
return {*}
"""


def sigma2_unbias(n1, n2, s1, s2):
    sigma2 = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return sigma2


"""
description: 计算t统计量
param {float/integer} n1: the number of array1
param {float/integer} n2: the number of array2
param {float} m1: mean of array1
param {float} m2: mean of array2
param {float} sigma: unbiased estimation of two arrays
return {*}
"""


def t_cal(n1, n2, m1, m2, sigma):
    t = (m1 - m2) / sigma / np.sqrt(1 / n1 + 1 / n2)
    return t


"""
description: 计算平均值和标准差（仅限xarray变量）
param {*} x: array
param {str} dim: 指定计算的维度名称
return {*}
"""


def fund_cal(x, dim):
    s = x.std(dim=dim, skipna=True)
    m = x.mean(dim=dim, skipna=True)
    return s, m


"""
description: 计算滑动t检验的t统计量以及对应置信度水平下的t临界值（已进行边界平滑处理）
param {*} da: 需要滑动的序列
param {str} dim: 需要滑动的维度名
param {integer} window: 滑动t检验基准点前后两子序列的长度
param {float} clevel: t临界值对应的置信度（如0.95）
return {*}
"""


def rolling_t_test(da, dim, window, clevel):
    from scipy.stats import t

    da = da.dropna(dim=dim)
    n = da.coords[dim].shape[0]
    n_dim = da.coords[dim]
    #   对边界进行了处理，在左右边界处，n1 != n2
    #   ti是最终输出的t统计量，为了和da保持相同的长度，将无法计算的边界部分设置为np.nan
    ti = np.zeros(n)
    ti[0:2] = np.nan
    ti[-1] = np.nan
    #   tlim是指定confidence level情况下的t临界值
    tlim = np.zeros(n)
    tlim[0:2] = np.nan
    ti[-1] = np.nan
    tlim[-1] = np.nan
    #   先计算左右边界
    for li in np.arange(2, window):
        #   左边界
        x1 = da[0:li]
        x2 = da[li : li + window]
        n1 = li
        n2 = window
        s1, m1 = fund_cal(x1, dim)
        s2, m2 = fund_cal(x2, dim)
        s_tmp = sigma2_unbias(n1, n2, s1, s2)
        t_tmp = t_cal(n1, n2, m1, m2, s_tmp)
        ti[li] = t_tmp
        tlim[li] = t.ppf(0.5 + 0.5 * clevel, n1 + n2 - 2)

        #   右边界
        x1 = da[-li - window : -li]
        x2 = da[-li:]
        n1 = window
        n2 = li
        s1, m1 = fund_cal(x1, dim)
        s2, m2 = fund_cal(x2, dim)
        s_tmp = sigma2_unbias(n1, n2, s1, s2)
        t_tmp = t_cal(n1, n2, m1, m2, s_tmp)
        ti[-li] = t_tmp
        tlim[-li] = t.ppf(0.5 + 0.5 * clevel, n1 + n2 - 2)

    #   中间段 n1=n2=window
    for i in np.arange(0, n - 2 * window + 1):
        x1 = da[i : i + window]
        x2 = da[i + window : i + 2 * window]
        n1 = window
        n2 = window
        s1, m1 = fund_cal(x1, dim)
        s2, m2 = fund_cal(x2, dim)
        s_tmp = sigma2_unbias(n1, n2, s1, s2)
        t_tmp = t_cal(n1, n2, m1, m2, s_tmp)
        ti[i + window] = t_tmp
        #   calculate tlim
        tlim[i + window] = t.ppf(0.5 + 0.5 * clevel, n1 + n2 - 2)

    del (n, li, x1, x2, n1, n2, s1, s2, m1, m2, s_tmp, t_tmp, i)
    t_test = xr.DataArray(ti, coords=[n_dim], dims=["time"])
    t_lim = xr.DataArray(tlim, coords=[n_dim], dims=["time"])
    return t_test, t_lim


"""
description: 计算有效自由度
            calculation of effective number of freedom
param {array} x
param {array} y: only used in way"2"
param {str} way:"0": for Leith(1973) method;
                "1": for Bretherton(1999) method;
                "2": for correlation coeffitient calculation (this method is introduced in class)
param {integer} l: for way"1", l should be 1 or 2, which means the times of lag correlation calculation;
                    for way"2", l should be less than len(x) - 2; It means the times of lead-lag correlation calculation
return {integer} neff
"""


def eff_DOF(x, y, way, l):
    num = np.shape(x)[0]
    #   calculate effective degree of freedom number with Leith(1973) method
    if way == "0":
        A1_tmp = x[:-1]
        A2_tmp = x[1:]
        tau = stats.linregress(A1_tmp, A2_tmp)[2]
        neff = int(-0.5 * np.log(tau) * num)
        del (num, A1_tmp, A2_tmp, tau)
    #   calculate with Bretherton(1999) method
    elif way == "1":
        if l == 1:
            #   calculate the one order ESS for x array
            A1_tmp = x[:-1]
            A2_tmp = x[1:]
            tau = stats.linregress(A1_tmp, A2_tmp)[2]
            neff = int(num * (1 - tau) / (1 + tau))
            print(neff)
            del (A1_tmp, A2_tmp, tau)
        elif l == 2:
            #   calculate the two order ESS for x&y array
            tau = np.ones(2)
            A1_tmp = x[:-1]
            A2_tmp = x[1:]
            tau[0] = stats.linregress(A1_tmp, A2_tmp)[2]

            B1_tmp = y[:-1]
            B2_tmp = y[1:]
            tau[1] = stats.linregress(B1_tmp, B2_tmp)[2]

            neff = int(num * (1 - tau[0] * tau[1]) / (1.0 + tau[0] * tau[1]))
            print(neff)
            del (A1_tmp, A2_tmp, B1_tmp, B2_tmp, tau)
    #   the calculation method of effective DOF in the calculation of correlation coefficient
    #   if using this method, then should import two data arrays that are used to calculate the correlation coefficient
    elif way == "2":
        #   calculate the lead-lag correlation of this two array
        tau = 0.0
        #   lead
        for i in np.arange(-l, 0, 1):
            A1_tmp = x[:i]
            A2_tmp = x[-i:]
            B1_tmp = y[:i]
            B2_tmp = y[-i:]
            tmp1 = stats.linregress(A1_tmp, A2_tmp)[2]
            tmp2 = stats.linregress(B1_tmp, B2_tmp)[2]
            # print(tmp1, tmp2)
            tau += tmp1 * tmp2
        #   lag
        for i in np.arange(1, l + 1, 1):
            A1_tmp = x[i:]
            A2_tmp = x[:-i]
            B1_tmp = y[i:]
            B2_tmp = y[:-i]
            tmp1 = stats.linregress(A1_tmp, A2_tmp)[2]
            tmp2 = stats.linregress(B1_tmp, B2_tmp)[2]
            # print(tmp1, tmp2)
            tau += tmp1 * tmp2
        #   contemporaneous correlation
        tau += 1.0

        neff = int(num / tau) - 2
        print("tau = ", tau, "neff = ", neff)
        del (num, A1_tmp, A2_tmp, B1_tmp, B2_tmp, tmp1, tmp2, tau)
    return neff


def cal_rlim(talpha, n):
    rlim = np.sqrt(talpha ** 2 / (n - 2.0 + talpha ** 2))
    return rlim


"""
description: 用于计算3D(season, year, lat, lon)变量场的超前滞后相关，同时将输出对应置信度的r临界值，其中x、y为已经处理为（季节x年份）的数据， freq="season"， ll表示超前（滞后）自身多少个时间单位， inan为在多余的表格中填充np.nan, clevel[0]为是否需要计算有效自由度, clevel[1]为置信度水平
param {*} x
param {*} y
param {str} freq: "season"
param {integer} ll
param {bool} inan
param {list} clevel
return {*}
"""


def leadlag_reg3D(x, y, freq, ll, inan, clevel):
    try:
        x.transpose("season", "time", ...)
        y.transpose("season", "time", ...)
        x_season = x.coords["season"]
        y_season = y.coords["season"]
        x_nseason = x_season.shape[0]
        y_nseason = y_season.shape[0]
        nlat = y.coords["lat"].shape[0]
        nlon = y.coords["lon"].shape[0]
        nyear = x.coords["time"].shape[0]
        if freq == "leadlag":
            if ll <= y_nseason:
                raise ValueError(
                    "ll should be less equal to the number of season in y}"
                )
            # elif ll == y_nseason:
            #     tmp = y_nseason
            # elif ll < y_nseason:
            #     tmp = ll
            avalue = np.zeros(
                (x_nseason, 2 * ll + y_nseason, nlat, nlon), dtype=np.float64
            )
            bvalue = np.zeros(
                (x_nseason, 2 * ll + y_nseason, nlat, nlon), dtype=np.float64
            )
            rvalue = np.zeros(
                (x_nseason, 2 * ll + y_nseason, nlat, nlon), dtype=np.float64
            )
            pvalue = np.zeros(
                (x_nseason, 2 * ll + y_nseason, nlat, nlon), dtype=np.float64
            )
            hyvalue = np.zeros(
                (x_nseason, 2 * ll + y_nseason, nlat, nlon), dtype=np.float64
            )
            reff = np.zeros(
                (x_nseason, 2 * ll + y_nseason, nlat, nlon), dtype=np.float64
            )
            tmp_time = np.arange(1, nyear, 1)
            for xsea in np.arange(0, x_nseason, 1):
                # calculate the lead-lag correlation of present year
                for ysea in np.arange(0, ll, 1):
                    (
                        avalue[xsea, ll + ysea, :, :],
                        bvalue[xsea, ll + ysea, :, :],
                        rvalue[xsea, ll + ysea, :, :],
                        pvalue[xsea, ll + ysea, :, :],
                        hyvalue[xsea, ll + ysea, :, :],
                    ) = dim_linregress(x[xsea, :], y[ysea, :, :, :])
                    if clevel[0] == True:
                        neff = eff_DOF(x[xsea, :], y[ysea, :, :, :], "1", 2)
                    elif clevel[0] == False:
                        neff = np.shape(x[xsea, :])[0]
                    t_lim = t.ppf(0.5 + 0.5 * clevel[1], neff)
                    reff[xsea, ll + ysea, :, :] = cal_rlim(t_lim, neff)

                    # calculate the lead correlation of last year
                    x_tmp = x[:, 1:]
                    y_tmp = y[:, :-1, :, :]
                    x_tmp.coords["time"], y_tmp.coords["time"] = tmp_time, tmp_time
                    (
                        avalue[xsea, ysea, :, :],
                        bvalue[xsea, ysea, :, :],
                        rvalue[xsea, ysea, :, :],
                        pvalue[xsea, ysea, :, :],
                        hyvalue[xsea, ysea, :, :],
                    ) = dim_linregress(x_tmp[xsea, :], y_tmp[ysea, :, :, :])
                    if clevel[0] == True:
                        neff = eff_DOF(x_tmp[xsea, :], y_tmp[ysea, :, :, :], "1", 2)
                    elif clevel[0] == False:
                        neff = np.shape(x_tmp[xsea, :])[0]
                    t_lim = t.ppf(0.5 + 0.5 * clevel[1], neff)
                    reff[xsea, ysea, :, :] = cal_rlim(t_lim, neff)

                    # calculate the lag correlation of next year
                    x_tmp = x[:, :-1]
                    y_tmp = y[:, 1:, :, :]
                    x_tmp.coords["time"], y_tmp.coords["time"] = tmp_time, tmp_time
                    (
                        avalue[xsea, 2 * ll + ysea, :, :],
                        bvalue[xsea, 2 * ll + ysea, :, :],
                        rvalue[xsea, 2 * ll + ysea, :, :],
                        pvalue[xsea, 2 * ll + ysea, :, :],
                        hyvalue[xsea, 2 * ll + ysea, :, :],
                    ) = dim_linregress(x_tmp[xsea, :], y_tmp[ysea, :, :, :])
                    if clevel[0] == True:
                        neff = eff_DOF(x_tmp[xsea, :], y_tmp[ysea, :, :, :], "1", 2)
                    elif clevel[0] == False:
                        neff = np.shape(x_tmp[xsea, :])[0]
                    t_lim = t.ppf(0.5 + 0.5 * clevel[1], neff)
                    reff[xsea, 2 * ll + ysea, :, :] = cal_rlim(t_lim, neff)

            if inan == True:
                for iiinan in np.arange(0, x_nseason - 1, 1):
                    (
                        avalue[iiinan, iiinan - ll + 1 :, :, :],
                        bvalue[iiinan, iiinan - ll + 1 :, :, :],
                        rvalue[iiinan, iiinan - ll + 1 :, :, :],
                        pvalue[iiinan, iiinan - ll + 1 :, :, :],
                        hyvalue[iiinan, iiinan - ll + 1 :, :, :],
                        reff[iiinan, iiinan - ll + 1 :, :, :],
                    ) = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

                    (
                        avalue[iiinan + 1, : iiinan + 1 :, :, :],
                        bvalue[iiinan + 1, : iiinan + 1 :, :, :],
                        rvalue[iiinan + 1, : iiinan + 1 :, :, :],
                        pvalue[iiinan + 1, : iiinan + 1 :, :, :],
                        hyvalue[iiinan + 1, : iiinan + 1 :, :, :],
                        reff[iiinan + 1, : iiinan + 1 :, :, :],
                    ) = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        elif freq == "lag":
            # elif ll == y_nseason:
            #     tmp = y_nseason
            # elif ll < y_nseason:
            #     tmp = ll
            avalue = np.zeros((x_nseason, ll + y_nseason, nlat, nlon), dtype=np.float64)
            bvalue = np.zeros((x_nseason, ll + y_nseason, nlat, nlon), dtype=np.float64)
            rvalue = np.zeros((x_nseason, ll + y_nseason, nlat, nlon), dtype=np.float64)
            pvalue = np.zeros((x_nseason, ll + y_nseason, nlat, nlon), dtype=np.float64)
            hyvalue = np.zeros(
                (x_nseason, ll + y_nseason, nlat, nlon), dtype=np.float64
            )
            reff = np.zeros((x_nseason, ll + y_nseason, nlat, nlon), dtype=np.float64)
            tmp_time = np.arange(1, nyear, 1)
            if inan == True:
                (
                    avalue[:, :, :, :],
                    bvalue[:, :, :, :],
                    rvalue[:, :, :, :],
                    pvalue[:, :, :, :],
                    hyvalue[:, :, :, :],
                    reff[:, :, :, :],
                ) = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

            for xsea in np.arange(0, x_nseason, 1):
                for ysea in np.arange(xsea, ll, 1):
                    # calculate the lead correlation of present year
                    x_tmp = x[:, :]
                    y_tmp = y[:, :, :, :]
                    (
                        avalue[xsea, ysea, :, :],
                        bvalue[xsea, ysea, :, :],
                        rvalue[xsea, ysea, :, :],
                        pvalue[xsea, ysea, :, :],
                        hyvalue[xsea, ysea, :, :],
                    ) = dim_linregress(x_tmp[xsea, :], y_tmp[ysea, :, :, :])
                    if clevel[0] == True:
                        neff = eff_DOF(x_tmp[xsea, :], y_tmp[ysea, :, :, :], "1", 2)
                    elif clevel[0] == False:
                        neff = np.shape(x_tmp[xsea, :])[0]
                    t_lim = t.ppf(0.5 + 0.5 * clevel[1], neff)
                    reff[xsea, ysea, :, :] = cal_rlim(t_lim, neff)

                for ysea in np.arange(0, ll, 1):
                    # calculate the lag correlation of next year
                    x_tmp = x[:, :-1]
                    y_tmp = y[:, 1:, :, :]
                    x_tmp.coords["time"], y_tmp.coords["time"] = tmp_time, tmp_time
                    (
                        avalue[xsea, ll + ysea, :, :],
                        bvalue[xsea, ll + ysea, :, :],
                        rvalue[xsea, ll + ysea, :, :],
                        pvalue[xsea, ll + ysea, :, :],
                        hyvalue[xsea, ll + ysea, :, :],
                    ) = dim_linregress(x_tmp[xsea, :], y_tmp[ysea, :, :, :])
                    if clevel[0] == True:
                        neff = eff_DOF(x_tmp[xsea, :], y_tmp[ysea, :, :, :], "1", 2)
                    elif clevel[0] == False:
                        neff = np.shape(x_tmp[xsea, :])[0]
                    t_lim = t.ppf(0.5 + 0.5 * clevel[1], neff)
                    reff[xsea, ll + ysea, :, :] = cal_rlim(t_lim, neff)
        elif freq == "present":
            avalue = np.zeros((x_nseason, y_nseason, nlat, nlon), dtype=np.float64)
            bvalue = np.zeros((x_nseason, y_nseason, nlat, nlon), dtype=np.float64)
            rvalue = np.zeros((x_nseason, y_nseason, nlat, nlon), dtype=np.float64)
            pvalue = np.zeros((x_nseason, y_nseason, nlat, nlon), dtype=np.float64)
            hyvalue = np.zeros(
                (x_nseason, y_nseason, nlat, nlon), dtype=np.float64
            )
            reff = np.zeros((x_nseason, y_nseason, nlat, nlon), dtype=np.float64)
            tmp_time = np.arange(1, nyear, 1)
            if inan == True:
                (
                    avalue[:, :, :, :],
                    bvalue[:, :, :, :],
                    rvalue[:, :, :, :],
                    pvalue[:, :, :, :],
                    hyvalue[:, :, :, :],
                    reff[:, :, :, :],
                ) = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

            for xsea in np.arange(0, x_nseason, 1):
                for ysea in np.arange(xsea, ll, 1):
                    # calculate the lead correlation of present year
                    x_tmp = x[:, :]
                    y_tmp = y[:, :, :, :]
                    (
                        avalue[xsea, ysea, :, :],
                        bvalue[xsea, ysea, :, :],
                        rvalue[xsea, ysea, :, :],
                        pvalue[xsea, ysea, :, :],
                        hyvalue[xsea, ysea, :, :],
                    ) = dim_linregress(x_tmp[xsea, :], y_tmp[ysea, :, :, :])
                    if clevel[0] == True:
                        neff = eff_DOF(x_tmp[xsea, :], y_tmp[ysea, :, :, :], "1", 2)
                    elif clevel[0] == False:
                        neff = np.shape(x_tmp[xsea, :])[0]
                    t_lim = t.ppf(0.5 + 0.5 * clevel[1], neff)
                    reff[xsea, ysea, :, :] = cal_rlim(t_lim, neff)
        else:
            raise ValueError(r"freq should be one of {season, year}")
    except ValueError as e:
        print(repr(e))
    del (
        x_season,
        y_season,
        x_nseason,
        y_nseason,
        nyear,
        xsea,
        ysea,
        x_tmp,
        y_tmp,
        tmp_time,
        neff,
    )
    return (avalue, bvalue, rvalue, pvalue, hyvalue, reff)

'''
description: 计算u、v风的显著性检验，当u或者v某一个分量通过检验时，认为该点的风场通过检验，最后的结果大于0的部分为通过检验的部分
param {*} u 待检验的u
param {*} v 待检验的v
param {*} ulim  u的检验标准
param {*} vlim  v的检验标准
return {*}
'''
def wind_check(u, v, ulim, vlim):
    u_index = u.where(abs(u) >= ulim)
    v_index = v.where(abs(v) >= vlim)
    tmp_a = u_index.fillna(0)
    tmp_b = v_index.fillna(0)
    new_index = tmp_a + tmp_b
    return(new_index)
    del(u_index, v_index, new_index, tmp_a, tmp_b)
    
    
def year_choose(year, da):
    selcase = da.sel(time=da.coords['time'].dt.year==year[0])
    for i in year[1:]:
        a = da.sel(time=da.coords['time'].dt.year==i)
        selcase = xr.concat([selcase, a], dim='time')
    return selcase


def generate_tmask(da1, da2, clevel):
    n1 = len(da1.coords['time'])
    n2 = len(da2.coords['time'])
    m1 = da1.mean(dim="time", skipna=True)
    m2 = da2.mean(dim="time", skipna=True)
    s1 = da1.std(dim="time", skipna=True)
    s2 = da2.std(dim="time", skipna=True)
    t_n = t_cal(n1, n2, m1, m2, sigma2_unbias(n1, n2, s1, s2))
    fdm = int(n1+n2-2)
    talpha = t.ppf(0.5+0.5*clevel, fdm)
    mask = xr.where(abs(t_n)>=talpha, 1.0, 0.0)
    
    return mask
    del n1,n2,m1,m2,s1,s2,fdn,talpha

# %%
# lat = [0.0, 1.0]
# lon = [0.0, 1.0]
# level = [850, 200]
# time = np.arange(1, 9, 1)

# x_n = np.reshape(x, (4, 2, 2, 2, 2), order="F")
# print(x[:, 0, 0, 0])
# print(x_n[1, :, 0, 0, 0])
# %%
