'''
Author: ChenHJ
Date: 2022-03-02 16:58:52
LastEditors: ChenHJ
LastEditTime: 2023-01-16 21:54:08
FilePath: /self_def/cal.py
Aim: 
Mission: 
目前已有的tag: 统计检验、显著性检验、生成mask
'''
# %%
import numpy as np
import xarray as xr
import os
import re
from cdo import *
import shutil
import pandas as pd
import sys

cdo = Cdo

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
from scipy import signal
from eofs.multivariate.standard import MultivariateEof
from eofs.standard import Eof
import metpy.calc as mpcalc
import metpy.constants as constants
import geocat.comp
from windspharm.xarray import VectorWind
import statsmodels.api as sm

# md:变量前处理类函数
def p_time(data, mon_s, mon_end, meanon=True):
    """该函数用于选取数据中每一年的相应月份，并可选地计算每一年这些相应月份的平均值；
    例如: 
    p_time(data, 6, 8, False)将选取数据中每一年的JJA，但不计算每年JJA的平均值；
    p_time(data, 6, 8, True)则在选取数据中每一年的JJA后计算每年JJA的平均值；

    Args:
        data (dataarray): 被用于选取数据的数据
        mon_s (int): 开始月份
        mon_end (int): 结束月份
        meanon (bool, optional): 是否要对选择的月份进行平均. Defaults to True.

    Returns:
        dataarray: 被挑选后的数据
    """    
    n_data = data.sel(
        time=(data.time.dt.month <= mon_end) * (data.time.dt.month >= mon_s)
    )
    n_mon = mon_end - mon_s + 1
    if meanon:
        n_data_mean = n_data.coarsen(time=n_mon).mean()
        return n_data_mean
    elif meanon == False:
        return n_data
    else:
        print("Bad argument: meanon")

def p_month(data, mon_s, mon_e):
    """本函数用于将需要的月份挑选出来，并存储为月份x年份x其他维度的形式

    Args:
        data (dataarray): 原数据
        mon_s (int): 开始月份
        mon_e (int): 结束月份

    Returns:
        dataarray: 月份x年份x其他维度的形式
    """    
    time = data["time"]
    data = data.transpose("time", ...).copy()
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

def standardize(da, dim="time"):
    """对原数据进行标准化处理

    Args:
        da (dataarray): 原数据
        dim (str, optional): 进行标准化的维度. Defaults to "time".

    Returns:
        dataarray: 标准化后的数据
    """    
    mean = da.mean(dim=dim, skipna=True)
    std = da.std(dim=dim, skipna=True)
    return (da - mean) / std

def mon_to_season(data, continuous=False, season="all"):
    """用于将月数据转换为季节数据，DJF已考虑过渡年，因此作为损失，最后一年将只保留1、2月用于计算倒数第二年的DJF；

    Args:
        data (dataarray): 原数据(monthly)
        continuous (bool, optional): 为True时，返回季节连续的数据，即时间维度为春、夏、秋、冬的连续数据. Defaults to False.
        season (str, optional): 返回的季节，为all时，分别返回MAM、JJA、SON、DJF的数据；为MAM/JJA/SON/DJF时，则返回对应季节的数据. Defaults to "all".

    Raises:
        ValueError: 参数传递错误；[continuous:True/False] [season:"all"/"MAM"/"JJA"/"SON"/"DJF"]

    Returns:
        dataarray: 处理后的季节数据，数量多少由season参数决定[season="all"，返回四个数据，否则为单个数据]
    """    
    new_data = data.sel(time=data.time[2:-1]).coarsen(time=3).mean()
    new_data.coords["time"] = pd.date_range(str(data.time.dt.year[0].data)+"0401", periods=len(data.time)/3-1, freq="3MS")
    data_MAM = new_data.sel(time=new_data.time.dt.month == 4)
    data_JJA = new_data.sel(time=new_data.time.dt.month == 7)
    data_SON = new_data.sel(time=new_data.time.dt.month == 10)
    data_DJF = new_data.sel(time=new_data.time.dt.month == 1)
    try:
        if continuous:
            return new_data
        elif season == "all":
            return data_MAM, data_JJA, data_SON, data_DJF
        elif season == "MAM":
            return data_MAM
        elif season == "JJA":
            return data_JJA
        elif season == "SON":
            return data_SON
        elif season == "DJF":
            return data_DJF
        else:
            raise ValueError('key arguments options:[continuous:True/False] [season:"all"/"MAM"/"JJA"/"SON"/"DJF"]')
    except ValueError as e:
        print("error: ", repr(e))
    del (new_data, data_MAM, data_JJA, data_SON, data_DJF)

def mon_to_season1D(da):
    """用于将1D的月数据转换为季节数据，DJF已考虑过渡年，因此作为损失，最后一年将只保留1、2月用于计算倒数第二年的DJF；输入的数据，可以以2月份作为时间结尾，也可以以12月份作为数据结尾；
        本函数仅保留用于旧的计算代码，请使用新函数：mon_to_season
        
    Args:
        da (dataarray): 原数据(monthly)

    Raises:
        ValueError: 输入数据的结尾出错

    Returns:
        dataarray: 季节数据
    """    
    da = da.transpose("time", ...)
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

def mon_to_season3D(da):
    """用于将3D(time x lat x lon)的月数据转换为季节数据，DJF已考虑过渡年，因此作为损失，最后一年将只保留1、2月用于计算倒数第二年的DJF；输入的数据，可以以2月份作为时间结尾，也可以以12月份作为数据结尾；
        本函数仅保留用于旧的计算代码，请使用新函数：mon_to_season
        
    Args:
        da (dataarray): 原数据(monthly)

    Raises:
        ValueError: 输入数据的结尾出错

    Returns:
        dataarray: 季节数据
    """ 
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

def mon_to_season4D(da):
    """用于将4D(time x level x lat x lon)的月数据转换为季节数据，DJF已考虑过渡年，因此作为损失，最后一年将只保留1、2月用于计算倒数第二年的DJF；输入的数据，可以以2月份作为时间结尾，也可以以12月份作为数据结尾；
        本函数仅保留用于旧的计算代码，请使用新函数：mon_to_season
        
    Args:
        da (dataarray): 原数据(monthly)

    Raises:
        ValueError: 输入数据的结尾出错

    Returns:
        dataarray: 季节数据
    """ 
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


def year_choose(year, da):
    """挑选数据中所需要的年份，并返回这些年份的集合

    Args:
        year (list): 需要的年份序列
        da (dataarray): 原数据

    Returns:
        dataarray: 相应年份的数据集合
    """    
    selcase = da.sel(time=da.coords["time"].dt.year == year[0])
    for i in year[1:]:
        a = da.sel(time=da.coords["time"].dt.year == i)
        selcase = xr.concat([selcase, a], dim="time")
    return selcase


def cal_lat_weighted_mean(da):
    """计算数据的纬度加权平均

    Args:
        da (dataarray): 原数据

    Returns:
        dataarray: 纬度加权平均结果
    """    
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    da_mean = da.weighted(weights).mean(("lat"), skipna=True)
    return da_mean
    
def cal_area_weighted_mean(da):
    """计算数据的区域（纬度加权）平均

    Args:
        da (dataarray): 原数据

    Returns:
        dataarray: 区域平均结果
    """    
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    da_mean = da.weighted(weights).mean(("lat"), skipna=True)
    da_mean = da_mean.mean(dim="lon", skipna=True)
    return da_mean


def deannual_cycle_for_monthdata_3D(da, timestart):
    """对于timexlatxlon的三维月数据去除年循环

    Args:
        da (dataarray): 原数据（monthly）
        timestart (str): 数据的起始时间

    Returns:
        dataarray: 去除年循环后结果
    """    
    da_month = p_month(da, 1, 12)
    da_monthmean = da_month.mean(dim="time", skipna=True)
    daa_month = da_month - da_monthmean
    daa_tmp = np.reshape(np.array(daa_month), [len(da.time), len(da.coords['lat']), len(da.coords['lon'])], order="F")
    lon = daa_month.coords["lon"]
    time_new = pd.date_range(timestart, periods=len(da.coords["time"]), freq="MS")
    daa = xr.DataArray(
        daa_tmp, coords={"time": time_new, "lat": da.coords["lat"], "lon": da.coords["lon"]}, dims=["time", "lat", "lon"]
    )
    return daa


def butterworth_filter(da, ya, yb, btype, deg=8):
    """本函数用于对xarray数组进行butterworth滤波

    Args:
        da (dataarray): 原数据
        ya (int): 滤波的时间年
        yb (int): 滤波的时间年（>ya）（仅带通时需要）
        btype (str): 滤波的类型: "bandpass", "lowpass" or "highpass"
        deg (int, optional): butterworth的阶数. Default to 8.

    Returns:
        dataarray: 滤波后结果
    """    
    da = da.transpose("time", ...)
    da = da - da.mean(dim="time", skipna=True)
    fs = 1.0
    highfreq = 2.0/ya/fs
    if btype == "bandpass":
        lowfreq = 2.0/yb/fs
        b, a = signal.butter(deg, [lowfreq, highfreq], btype=btype, fs=fs)
    else:
        b, a = signal.butter(deg, highfreq, btype=btype, fs=fs)
    new_data = da
    new_data.data = signal.filtfilt(b, a, da, axis=0, padtype="even")
    return new_data

def dezonal_mean(da):
    """计算纬偏

    Args:
        da (dataarray): 原数据

    Returns:
        dataarray: 纬偏值
    """    
    da_zonal = da.mean(dim="lat", skipna=True)
    return da - da_zonal

def filplonlat(ds): 
    """对经度以0至360排列的数据进行重排至-180至180

    Args:
        ds (dataarray): 原数据

    Returns:
        dataarray: 重排后数据
    """    
    # To facilitate data subsetting 
    print(ds.attrs) 
    print( f'\n\nBefore flip, lon range is [{ds["lon"].min().data}, {ds["lon"].max().data}].' ) 
    ds["lon"] = ((ds["lon"] + 180) % 360) - 180 
    # Sort lons, so that subset operations end up being simpler. 
    ds = ds.sortby("lon") 
    print( f'\n\nAfter flip, lon range is [{ds["lon"].min().data}, {ds["lon"].max().data}].' ) 
    # To facilitate data subsetting 
    ds = ds.sortby("lat", ascending=True) 
    print(ds.attrs) 
    # print_debug('\n\nAfter sorting lat values, ds["lat"] is:') 
    # print_debug(ds["lat"]) 
    return ds

# md:数据预处理（文件层面）
def p_year(srcPath, dstPath, start, end):
    """挑选数据指定时间段

    Args:
        srcPath (str): 原始数据保存路径
        dstPath (str): 目标数据保存路径
        start (int): 开始年份
        end (int): 结束年份
    """    
    g = os.walk(srcPath)
    for path, dir_list, file_list in g:
        for file_name in file_list:
            inputfile = os.path.join(srcPath, file_name)
            outputfile = os.path.join(dstPath, file_name[:-16] + str(start) + "01-" + str(end) + "12.nc")
            cdo.selyear(str(start) + r"/" + str(end), input=inputfile, output=outputfile)


def CMIP6_predealing_1(srcPath, tmpPath, dstPath, variable, freq, rl):
    """CMIP6数据预处理函数，包括数据插值和数据合并

    Args:
        srcPath (str): 原始数据保存路径
        tmpPath (str): 存放插值后数据的路径
        dstPath (str): 目标数据保存路径
        variable (str): 变量名
        freq (str): 数据的frequency（Amon或day等）
        rl (str): realization name（r1i1p1f1等）
    """    
    if os.path.exists(tmpPath) == True:
        shutil.rmtree(tmpPath)
        os.mkdir(tmpPath)
    else:
        os.mkdir(tmpPath)
    # length = len(regressRule)  #len()函数返回对象(字符、列表、元组等)长度或项目的个数，在此处为返回regressRule的元素个数
    # for path,dir_list,file_list in g:    #for的遍历，os.walk遍历了几次（文件夹本身1次，子文件夹有几个就加几次），for就遍历几次，其中子文件夹中的所有文件名都会记录为一个列表
    for var in variable:
        g = os.walk(srcPath)
        for path, dir_list, file_list in g:
            # for x in np.arange(0, len(exp)):
            #     for i in np.arange(0, len(regressRule)):  # 有多少个模式在regressRule里就遍历多少次
            inputString = ""
            mergelist = []
            for file_name in file_list:  # 此处对file_list中的元素进行遍历并逐一赋值给file_name
                if (
                    re.search(var + "_", file_name) != None
                    and re.search("_" + freq + "_", file_name) != None
                    and re.search(rl, file_name)
                ):
                    print(file_name)
                    inputfile = os.path.join(path, file_name)
                    outputfile = os.path.join(tmpPath, "tmp_" + file_name)
                    cdo.remapbil("r144x72", input=inputfile, output=outputfile)
                    mergelist.append(file_name)
                else:
                    pass
            if len(mergelist) == 0:
                print("There are no need to combine the model data ")
                pass
            elif len(mergelist) == 1:  # 如果该模式只有一个文件（可能是已经被处理好或本来就只有一个文件不需要进行合并）
                cdo.copy(
                    input=os.path.join(tmpPath, "tmp_" + mergelist[0]),
                    output=os.path.join(dstPath, mergelist[0]),
                )
            else:
                for j in np.arange(0, len(mergelist)):
                    inputString += os.path.join(
                        tmpPath, "tmp_" + mergelist[j] + " "
                    )  # 加字符串形式的空格是为了满足 cdo ifile1 ifile2……这样的格式
                cdo.mergetime(
                    input=inputString,
                    output=os.path.join(
                        dstPath, mergelist[0][:-11] + mergelist[j][-11:],
                    ),
                )

def CMIP6_check(dstPath, modelname, yearstart, yearend):
    """CMIP6文件检查

    Args:
        dstPath (str): 文件路径
        modelname (str): 模式名称
        yearstart (str): 开始年份
        yearend (str): 结束年份
    """    
    for name, end in zip(modelname, yearend):
        g = os.walk(dstPath)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                if re.search("_" + name + "_", file_name) != None:
                    newname = file_name[:-16] + yearstart + "-" + end + ".nc"
                    # print(newname)
                    os.rename(
                        os.path.join(path, file_name), os.path.join(path, newname)
                    )
                    file = xr.open_dataset(os.path.join(path, newname))
                    varname = newname[: newname.find("_")]
                    var = file[varname]
                    print(
                        "varname = ",
                        varname,
                        str(np.array(var.coords["time"][0]))[:7],
                        str(np.array(var.coords["time"][-1]))[:7],
                        "time_num = ",
                        len(var.coords["time"]),
                        "units = ",
                        var.attrs["units"],
                    )
                    del(file)


def uniform_timestamp(filepath, dstpath, var, start, end, periods):
    """统一数据的时间坐标

    Args:
        filepath (str): 文件路径
        dstpath (str): 保存路径
        var (str): 变量名
        start (str): 时间坐标开始时间
        end (str): 时间坐标结束时间
        periods (str): 时间坐标间隔
    """    
    time = pd.date_range(start, end, freq=periods)
    f = xr.open_dataset(filepath)
    fvar = f[var]
    fvar.coords['time'] = time
    fvar.to_netcdf(dstpath)
    

def uniform_plev(filepath, dstpath, var):
    """统一数据的垂直坐标

    Args:
        filepath (str): 文件路径
        dstpath (str): 保存路径
        var (str): 变量名
    """    
    plev = np.array([100000.0, 92500.0, 85000.0, 70000.0, 60000.0, 50000.0, 40000.0, 30000.0, 25000.0, 20000.0, 15000.0, 10000.0, 7000.0, 5000.0, 3000.0, 2000.0, 1000.0, 500.0, 100.0])
    f = xr.open_dataset(filepath)
    fvar = f[var]
    fvar.coords['plev'] = plev
    fvar.to_netcdf(dstpath)

# md:统计类——相关、回归计算
def detrend_dim(da, dim="time", deg=1, demean=True):      
    """对原数据进行去趋势；建议仅需要去趋势后结果时使用本函数。如果还需要趋势值及趋势值相关的p值，请使用new_detrend_dim

    Args:
        da (dataarray): 原数据
        dim (str, optional): 对哪一个维度去趋势. Defaults to "time".
        deg (int, optional): 回归拟合的次幂数，1为线性拟合（线性去趋势）. Defaults to 1.
        demean (bool, optional): 设置为False则返回去趋势但不去平均值（减去at）的数据，设置为True则返回既去除趋势又去除平均值的数据(减去at+b). Defaults to True.
    Returns:
        dataarray: 去趋势后的数据
    """    
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg, skipna=True)
    # regress = dim_linregress(np.arange(len(da.coords["time"])), da)
    # print("avalue = ", regress[0].data)
    # print("rvalue = ", regress[2].data)
    # print("pvalue = ", regress[3].data)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    if demean == False:
        return (da - fit) + da.mean(dim="time", skipna=True)
    elif demean == True:
        return da - fit

def dim_linregress(x, y, input_core_dims="time"):     
    """本函数用于计算回归或相关pattern，y=ax+b

    Args:
        x (dataarray): 自变量
        y (dataarray): 因变量
        input_core_dims (str, optional): 用于apply_ufunc不被广播的维度. Defaults to "time".

    Returns:
        dataarray: 斜率slope
        dataarray: 截距intercept
        dataarray: 相关系数rvalue
        dataarray: pvalue
        dataarray: hypothesis
    """    
    # returns: slope,intercept,rvalue,pvalue,hypothesis
    return xr.apply_ufunc(
        stats.linregress,
        x,
        y,
        input_core_dims=[[input_core_dims], [input_core_dims]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask="parallelized",
    )

def leadlag_reg(x, y, ll, clevel=[True, 0.95], inan=True, freq="season"):  
    """用于计算超前滞后相关，同时将输出对应置信度的r临界值，其中x、y为已经处理为（季节x年份）的数据

    Args:
        x (dataarray): 自变量（season*year）
        y (dataarray): 因变量（season*year）
        ll (int): 超前（滞后）自身多少个时间单位
        clevel (list, optional): clevel[0]为是否需要计算有效自由度, clevel[1]为置信度水平. Defaults to [True, 0.95].
        inan (bool, optional): 是否在多余的表格中填充np.nan. Defaults to True.
        freq (str, optional): 超前滞后的时间频次. Defaults to "season".

    Raises:
        ValueError: ll should be less equal to the number of season in y
        ValueError: freq should be one of {season, year}

    Returns:
        dataarray: 斜率slope
        dataarray: 截距intercept
        dataarray: 相关系数rvalue
        dataarray: pvalue
        dataarray: hypothesis
        dataarray: 对应置信度的r临界值
    """    
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
                    "ll should be less equal to the number of season in y"
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


def leadlag_reg_rolling(x, y, ll, window, inan=True, freq="season"):
    """用于计算滑动超前滞后相关，其中x、y为已经处理为（季节x年份）的数据

    Args:
        x (dataarray): 自变量（season*year）
        y (dataarray): 因变量（season*year）
        ll (int): 超前（滞后）自身多少个时间单位
        window (int): 滑动窗口的长度
        inan (bool, optional): 是否在多余的表格中填充np.nan. Defaults to True.
        freq (str, optional): 超前滞后的时间频次. Defaults to "season".

    Raises:
        ValueError: ll should be less equal to the number of season in y
        ValueError: freq should be one of {season, year}

    Returns:
        dataarray: 斜率slope
        dataarray: 截距intercept
        dataarray: 相关系数rvalue
        dataarray: pvalue
        dataarray: hypothesis
    """     
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


def leadlag_reg3D(x, y, ll, clevel=[True, 0.95], inan=True, freq="season"):
    """用于计算3D(season, year, lat, lon)变量场的超前滞后相关，同时将输出对应置信度的r临界值，其中x、y为已经处理为（季节x年份）的数据

    Args:
        x (dataarray): 自变量（season*year*lat*lon）
        y (dataarray): 因变量（season*year*lat*lon）
        ll (int): 超前（滞后）自身多少个时间单位
        clevel (list, optional): clevel[0]为是否需要计算有效自由度, clevel[1]为置信度水平. Defaults to [True, 0.95]
        inan (bool, optional): 是否在多余的表格中填充np.nan. Defaults to True.
        freq (str, optional): 超前滞后的时间频次. Defaults to "season".

    Raises:
        ValueError: ll should be less equal to the number of season in y
        ValueError: freq should be one of {season, year}

    Returns:
        dataarray: 斜率slope
        dataarray: 截距intercept
        dataarray: 相关系数rvalue
        dataarray: pvalue
        dataarray: hypothesis
        dataarray: 对应置信度的r临界值
    """  
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
            hyvalue = np.zeros((x_nseason, y_nseason, nlat, nlon), dtype=np.float64)
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

def rolling_reg_index(x, y, time, window, returndataset=False):
    """计算滑动回归，仅一维numpy数组

    Args:
        x (ndarray): 自变量（一维）
        y (ndarray): 因变量（一维）
        time (datetime): 时间序列，不能为dataarray
        window (int): 滑动窗口长度，建议为单数
        returndataset (bool, optional): 是否以dataset格式返回. Default to False.

    Returns:
        dataset/
        float: 回归系数
        float: 截距
        float: 相关系数
        float: pvalue
        float: hypothesis
    """    
    avalue = np.full(len(time), np.nan)
    bvalue = np.full(len(time), np.nan)
    rvalue = np.full(len(time), np.nan)
    pvalue = np.full(len(time), np.nan)
    hyvalue = np.full(len(time), np.nan)
    for nx in np.arange(window//2, len(time)-window//2):
        avalue[nx], bvalue[nx], rvalue[nx], pvalue[nx], hyvalue[nx] = stats.linregress(x[nx:nx+window], y[nx:nx+window])
    if returndataset == True:
        regress = xr.Dataset(
        data_vars=dict(
            avalue=(["time"], avalue),
            bvalue=(["time"], bvalue),
            rvalue=(["time"], rvalue),
            pvalue=(["time"], pvalue),
            hyvalue=(["time"], hyvalue),
            ),
        coords=dict(
            time=time,
        ),
        attrs=dict(
            description="rolling_regression"
        ))
        return regress
    elif returndataset == False:
        return avalue, bvalue, rvalue, pvalue, hyvalue

def rolling_regression_pattern(x, y, time, window):
    """计算滑动回归，利用apply_ufunc广播

    Args:
        x (dataarray): 自变量
        y (dataarray): 因变量
        time (datetime): 时间序列，不能为dataarray
        window (int): 滑动窗口长度，建议为单数

    Returns:
        dataarray: 回归系数
        dataarray: 截距
        dataarray: 相关系数
        dataarray: pvalue
        dataarray: hypothesis
    """    
    avalue, bvalue, rvalue, pvalue, hyvalue = xr.apply_ufunc(
        rolling_reg_index,
        x,
        y,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"],["time"],["time"],["time"],["time"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"time":time, "window": window, "returndataset": False},
        join="left"
        )
    avalue = avalue.transpose("time", ...)
    bvalue = bvalue.transpose("time", ...)
    rvalue = rvalue.transpose("time", ...)
    pvalue = pvalue.transpose("time", ...)
    hyvalue = hyvalue.transpose("time", ...)
    return avalue, bvalue, rvalue, pvalue, hyvalue

def cal_rdiff(r1,r2):
    """计算两相关系数之差

    Args:
        r1 (dataarray): 相关系数1
        r2 (dataarray): 相关系数2

    Returns:
        dataarray: 相关系数之差
    """    
    Z1 = np.arctanh(r1)
    Z2 = np.arctanh(r2)
    zdiff = Z1-Z2
    return np.tanh(zdiff)

def cal_rMME(r,dim):
    """计算相关系数集合平均

    Args:
        r (dataarray): 相关系数
        dim (str): 沿着哪一个维度进行集合平均

    Returns:
        dataarray: 相关系数集合平均
    """    
    return np.tanh(np.arctanh(r).mean(dim=dim,skipna=True))


def multi_var_regression(y, x_list, dim="time", concat_dim="new_dim"):
    """计算多元线性回归的回归系数（偏回归系数）及其p值

    Args:
        y (dataarray): 因变量
        x_list (list): 多元变量的列表，函数将自动把这些变量拼接起来
        dim (str, optional): 用于计算时间的维度名. Defaults to "time".
        concat_dim (str, optional): 拼接而成的新维度的名字. Defaults to "new_dim".
        
    Returns:
        dataarray: 偏回归系数
        dataarray: p值
    """    
    def func(y,x):
        func0 = lambda y, x: sm.OLS(y,sm.add_constant(x)).fit()
        return func0(y,x).params[1:], func0(y,x).pvalues[1:]
    return xr.apply_ufunc(
        func,
        y,
        xr.concat(x_list, dim=concat_dim),
        input_core_dims=[[dim], [dim, concat_dim]],
        output_core_dims=[[concat_dim], [concat_dim]],
        vectorize=True,
        dask="parallelized",
        keep_attrs=True,
    )
    
def new_detrend_dim(y, dim="time", demean=True):
    """新的去趋势函数，优势是同时可以返回趋势、趋势的p值、去趋势值

    Args:
        y (dataarray): 原数据
        dim (str, optional): 需要去趋势的维度. Defaults to "time".
        demean (bool, optional): 是否将去趋势前的平均值重新加回. Defaults to True.

    Returns:
        dataarray: 趋势
        dataarray: 趋势的p值
        dataarray: 去趋势值
    """    
    def func(y,x):
        func0 = lambda y, x: sm.OLS(y,sm.add_constant(x)).fit()
        return func0(y,x).params[:], func0(y,x).pvalues[:]
    cal_time = xr.DataArray(
        data=range(len(y.time)),
        dims=["time"],
        coords=dict(
            time=(["time"], y.time.data),
        )
    )
    polyval, pvalue = xr.apply_ufunc(
        func,
        y,
        cal_time,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[["degree"], ["degree"]],
        vectorize=True,
        dask="parallelized",
        keep_attrs=True,
        dask_gufunc_kwargs={"output_sizes":{"degree":2}}
    )
    polyval.coords["degree"] = np.array([0, 1])
    fit = xr.polyval(cal_time, polyval)
    if demean:
        return polyval, pvalue, (y - fit), 
    else:
        return (polyval, pvalue, y - fit) + y.mean(dim="time", skipna=True)

def cal_pcc(var1, var2):
    """计算两变量场之间的空间相关系数

    Args:
        var1 (dataarray): 原数据1
        var2 (dataarray): 原数据2

    Returns:
        float: 空间相关系数
    """    
    # var1.coords["level"].name = "plev"
    # var1 = var1.where(var2.isnull() == False)
    # print(var1)
    array1 = np.array(var1)
    array2 = np.array(var2)
    array1 = np.where(np.isnan(array2), np.nan, array1)
    array2 = np.where(np.isnan(array1), np.nan, array2)
    # array1 = array1
    lat = var1.coords["lat"]
    lon = var1.coords["lon"]
    # plev = var1.coords["level"]
    shape = np.shape(var1)
    shape_prod = np.prod(shape)
    # print(shape_prod)
    h = 2.5 / 180.0 * np.pi * 6371000.0
    s_up = np.array(
        2.5 / 360.0 * np.pi * 2.0 * 6371000.0 * np.cos((lat + 1.25) * np.pi / 180.0)
    )
    s_down = np.array(
        2.5 / 360.0 * np.pi * 2.0 * 6371000.0 * np.cos((lat - 1.25) * np.pi / 180.0)
    )
    # s_up = np.tile(
    #     np.array(
    #         2.5 / 360.0 * np.pi * 2.0 * 6371000.0 * np.cos((lat + 1.25) * np.pi / 180.0)
    #     ),
    #     int(shape_prod / len(lat)),
    # ).reshape(np.shape(var1))
    # print(np.shape(s_up))
    # s_down = np.tile(
    #     np.array(
    #         2.5 / 360.0 * np.pi * 2.0 * 6371000.0 * np.cos((lat - 1.25) * np.pi / 180.0)
    #     ),
    #     int(shape_prod / len(lat)),
    # ).reshape(np.shape(var1))

    ds = (s_up + s_down) / 2.0 * h
    s = np.sum(ds)
    # print(s)
    wei = ds / s
    # print(ds)
    weight = np.zeros(np.shape(var1))
    for i in np.arange(len(lat)):
        weight[i, :] = wei[i]
        # weight[i, :] = wei[i]
    # print(weight)
    # quanzhong = xr.DataArray(weight,coords=[plev, lat, lon], dims=['level','lat','lon'])
    # quanzhong.name = "weight"
    # quanzhong.to_netcdf("/mnt/c/Users/11529/Desktop/quanzhong.nc")
    v1v2 = np.nansum(array1 * array2 * weight, axis=(0, 1))
    # v1v2 = np.nansum(array1 * array2 * weight, axis=(0, 1))
    # print(np.shape(v1v2))
    v1v1 = np.nansum(array1 * array1 * weight, axis=(0, 1))
    # v1v1 = np.nansum(array1 * array1 * weight, axis=(0, 1))
    v2v2 = np.nansum(array2 * array2 * weight, axis=(0, 1))
    # v2v2 = np.nansum(array2 * array2 * weight, axis=(0, 1))
    # print(v2v2)
    pcc = v1v2 / np.sqrt(v1v1) / np.sqrt(v2v2)
    return pcc

# md:统计类——显著性检验类
def if_two_group_have_diff_mean_t(da1, da2, dim="time", clevel=0.95, return_mask=True):
    """用于比较两组数据之间，平均值是否具有组间差异

    Args:
        da1 (dataarray): 原数据1
        da2 (dataarray): 原数据2
        dim (str, optional): 对哪一维度进行检验. Defaults to "time".
        clevel (float, optional): 显著性水平. Defaults to 0.95.
        return_mask (bool, optional): 是否返回显著性检验后的mask，mask的1.0为通过显著性检验，0.0为未通过. Defaults to True.

    Returns:
        dataarray: mask或t值
    """    
    n1 = np.shape(da1.coords[dim])[0]
    n2 = np.shape(da2.coords[dim])[0]
    sigma = np.sqrt(((n1 - 1) * da1.std(dim=dim,skipna=True) ** 2 + (n2 - 1) * da2.std(dim=dim,skipna=True) ** 2) / (n1 + n2 - 2))
    tvalue = (da1.mean(dim=dim,skipna=True) - da2.mean(dim=dim,skipna=True))/sigma/np.sqrt(1/n1 + 1/n2)
    tlim = t.ppf(0.5 + 0.5 * clevel, n1 + n2 - 2)
    del(n1,n2,sigma)
    if return_mask:
        return xr.where(abs(tvalue)>=tlim, 1.0, 0.0)
    else:
        return tvalue

def rolling_t_test(da, window, dim="time", clevel=0.95, return_mask=True):
    """计算滑动t检验的t统计量以及对应置信度水平下的t值及其t临界值（已进行边界平滑处理）

    Args:
        da (dataarray): 原数据
        window (int): 滑动t检验基准点前后两子序列的长度
        dim (str, optional): 需要滑动的维度名. Defaults to "time".
        clevel (float, optional): t临界值对应的置信度（如0.95）. Defaults to 0.95.
        return_mask (bool, optional): 是否返回显著性检验后的mask，mask的1.0为通过显著性检验，0.0为未通过. Defaults to True.

    Returns:
        dataarray: mask或t值与t临界值
    """    
    da = da.dropna(dim=dim).transpose(dim, ...)
    n = da.coords[dim].shape[0]
    #   对边界进行了处理，在左右边界处，n1 != n2
    #   ti是最终输出的t统计量，为了和da保持相同的长度，将无法计算的边界部分设置为np.nan
    ti = da.copy()
    ti[0:2] = np.nan
    ti[-1] = np.nan
    #   tlim是指定confidence level情况下的t临界值
    tlim = da.copy()
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
        t_tmp = if_two_group_have_diff_mean_t(x1,x2,dim=dim,clevel=clevel,return_mask=False).data
        ti[li] = t_tmp
        tlim[li] = t.ppf(0.5 + 0.5 * clevel, n1 + n2 - 2)

        #   右边界
        x1 = da[-li - window : -li]
        x2 = da[-li:]
        n1 = window
        n2 = li
        t_tmp = if_two_group_have_diff_mean_t(x1,x2,dim=dim,clevel=clevel,return_mask=False).data
        ti[-li] = t_tmp
        tlim[-li] = t.ppf(0.5 + 0.5 * clevel, n1 + n2 - 2)

    #   中间段 n1=n2=window
    for i in np.arange(0, n - 2 * window + 1):
        x1 = da[i : i + window]
        x2 = da[i + window : i + 2 * window]
        n1 = window
        n2 = window
        t_tmp = if_two_group_have_diff_mean_t(x1,x2,dim=dim,clevel=clevel,return_mask=False).data
        ti[i + window] = t_tmp
        #   calculate tlim
        tlim[i + window] = t.ppf(0.5 + 0.5 * clevel, n1 + n2 - 2)

    del (n, li, x1, x2, n1, n2, t_tmp, i)
    if return_mask:
        return xr.where(abs(ti)>=tlim, 1.0, 0.0)
    else:
        return ti, tlim

def eff_DOF(x, y, way=0, l=1):
    """计算有效自由度

    Args:
        x (array): 原数据1
        y (array): 原数据2，仅当way=2时使用.
        
        way (int, optional): "0": for Leith(1973) method; "1": for Bretherton(1999) method; "2": for correlation coeffitient calculation (this method is introduced in class). Defaults to 0.
        l (int, optional): for way"1", l should be 1 or 2, which means the times of lag correlation calculation;
                            for way"2", l should be less than len(x) - 2; It means the times of lead-lag correlation calculation. Defaults to 1.

    Returns:
        int: 有效自由度
    """    
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

def cal_rlim(n, clevel=0.95):
    """计算相关系数临界值

    Args:
        n (int): 数据长度
        clevel (float, optional): 显著性水平. Defaults to 0.95.

    Returns:
        float: 相关系数临界值
    """    
    talpha = t.ppf(0.5 + 0.5 * clevel, n-2)
    rlim = np.sqrt(talpha ** 2 / (n-2.0 + talpha ** 2))
    return rlim

def wind_check(u, v, ulim, vlim):
    """计算u、v风的显著性检验，当u或者v某一个分量通过检验时，认为该点的风场通过检验，最后的结果大于0的部分为通过检验的部分

    Args:
        u (dataarray): 待检验的u
        v (dataarray): 待检验的v
        ulim (dataarray): u的检验标准
        vlim (dataarray): v的检验标准

    Returns:
        dataarray: mask
    """    
    u_index = u.where(abs(u) >= ulim)
    v_index = v.where(abs(v) >= vlim)
    tmp_a = u_index.fillna(0)
    tmp_b = v_index.fillna(0)
    new_index = tmp_a + tmp_b
    return new_index


def MME_reg_mask(MME_var, std_var, N, mask=True):
    """生成临界值或mask；对于mask，值1.0代表该点通过95%显著性水平检验

    Args:
        MME_var (dataarray): MME
        std_var (dataarray): 样本间标准差
        N (int): 样本数量
        mask (bool, optional): 是否生成mask. Default to True.

    Returns:
        dataarray: 临界值或mask
    """    
    lamb = 1.96
    lim = std_var*lamb/np.sqrt(N)
    if mask == True:
        gmask = xr.where(abs(MME_var)-lim>=0.0, 1.0, 0.0)
        return gmask
    elif mask == False:
        return lim
    

def cal_mean_bootstrap_confidence_intervals(da, B, alpha=0.95):
    """利用bootstrap方法生成平均值的置信区间

    Args:
        da (dataarray): 原数据
        B (int): 重复次数
        alpha (float, optional): 置信度水平. Default to 0.95.

    Returns:
        float: 置信区间下界
        float: 置信区间上界
    """    
    nparray = np.array(da)
    low_lim = int(B*(1-alpha)/2)
    high_lim = int(B*(1.0-(1-alpha)/2))
    tmp_mean = np.zeros(B)
    for i in range(B):
        bs_sample = np.random.choice(nparray, size=len(nparray))
        tmp_mean[i] = np.mean(bs_sample)
    sort_mean = np.sort(tmp_mean)
    return sort_mean[low_lim-1], sort_mean[high_lim-1]


def cal_median_bootstrap_confidence_intervals(da, B, alpha=0.95):
    """利用bootstrap方法生成中位数的置信区间

    Args:
        da (dataarray): 原数据
        B (int): 重复次数
        alpha (float, optional): 置信度水平. Default to 0.95.

    Returns:
        float: 置信区间下界
        float: 置信区间上界
    """ 
    nparray = np.array(da)
    low_lim = int(B*(1-alpha)/2)
    high_lim = int(B*(1.0-(1-alpha)/2))
    tmp_median = np.zeros(B)
    for i in range(B):
        bs_sample = np.random.choice(nparray, size=len(nparray))
        tmp_median[i] = np.median(bs_sample)
    sort_median = np.sort(tmp_median)
    return sort_median[low_lim-1], sort_median[high_lim-1]


def cal_mean_bootstrap_confidence_intervals_pattern(da, B, alpha=0.95, dim="models"):
    """利用bootstrap方法生成平均值的置信区间，利用xarray广播运算

    Args:
        da (dataarray): 原数据
        B (int): 重复次数
        alpha (float, optional): 置信度水平. Default to 0.95.
        dim (str, optional): 对哪一个维度进行bootstrap验证. Default to "models".

    Returns:
        float: 置信区间下界
        float: 置信区间上界
    """  
    low_lim, high_lim = xr.apply_ufunc(
        cal_mean_bootstrap_confidence_intervals,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[],[]],
        vectorize=True,
        dask="parallelized",
        kwargs={"B":B, "alpha":alpha}
        )
    low_lim.name = "low_lim"
    high_lim.name = "high_lim"
    return low_lim, high_lim

def generate_bootstrap_mask(low_lim1, high_lim1, low_lim2, high_lim2):
    """对于两组不同的数据，利用bootstrap方法各自计算得到的置信区间上下界确定两组均值存在显著差异的mask
        mask大于0.0处为两组均值存在显著差异

    Args:
        low_lim1 (dataarray): 原数据1得到的置信区间下界
        high_lim1 (dataarray): 原数据1得到的置信区间上界
        low_lim2 (dataarray): 原数据2得到的置信区间下界
        high_lim2 (dataarray): 原数据2得到的置信区间上界

    Returns:
        dataarray: mask
    """    
    tmp1 = xr.where(low_lim1-high_lim2>0, 1.0, 0.0)
    tmp2 = xr.where(low_lim2-high_lim1>0, 1.0, 0.0)
    tmp1.name = "mask"
    tmp2.name = "mask"
    tmp = tmp1 + tmp2
    return tmp

def Fisher_permutation_test(x1,y1,x2,y2,K=1000, axis=0, return_mask=False, CI=0.95):
    """利用Fisher置换检验计算两组因变量-自变量所计算得到的相关关系是否存在显著差异

    Args:
        x1 (dataarray): 自变量1
        y1 (dataarray): 因变量1
        x2 (dataarray): 自变量2
        y2 (dataarray): 因变量2
        K (int, optional): 置换检验的抽样次数. Defaults to 1000.
        axis (int, optional): 对哪一轴进行检验. Defaults to 0.
        return_mask (bool, optional): 是否返回mask. Defaults to False.
        CI (float, optional): 置信度水平. Defaults to 0.95.

    Returns:
        dataarray: mask
    """    
    #   calculate the original regression coefficients differences
    odiff = stats.linregress(x1.data,y1.data)[0]-stats.linregress(x2.data, y2.data)[0]
    #   combine the x1 and x2/y1 and y2
    x_combine = np.append(np.array(x1),np.array(x2),axis=axis)
    y_combine = np.append(np.array(y1), np.array(y2), axis=axis)
    #   resample the x_combine and y_combine
    fi_d = np.zeros(K)
    loc_num = range(0,len(x1)+len(x2))
    for test_i in range(K):
        fi_sample_list1 = np.random.choice(loc_num, size=13, replace=False)
        fi_sample_list2 = list(set(loc_num) ^ set(fi_sample_list1))
        #   generate te new samples
        fi_x_sample1 = [x_combine[i] for i in fi_sample_list1]
        fi_y_sample1 = [y_combine[i] for i in fi_sample_list1]
        fi_x_sample2 = [x_combine[i] for i in fi_sample_list2]
        fi_y_sample2 = [y_combine[i] for i in fi_sample_list2]
            
        fi_regress1 = stats.linregress(fi_x_sample1, fi_y_sample1)
        fi_regress2 = stats.linregress(fi_x_sample2, fi_y_sample2)
        d = fi_regress1[0]- fi_regress2[0]
        fi_d[test_i] = d
    fi_d = np.sort(fi_d)
    if return_mask:
        return np.where(min(sum(i > odiff for i in fi_d)/K, sum(i < odiff for i in fi_d)/K)*2<=1.-CI, 1.0, 0.0)
    else:
        return min(sum(i > odiff for i in fi_d)/K, sum(i < odiff for i in fi_d)/K)*2

def Fisher_Z_test(r1,r2,N1,N2,return_mask=False,CI=0.95):
    """利用Fisher-Z变换计算相关系数间是否存在差异

    Args:
        r1 (dataarray): 相关系数1
        r2 (dataarray): 相关系数2
        N1 (int): 样本长度1
        N2 (int): 样本长度2
        return_mask (bool, optional): 是否返回mask. Defaults to False.
        CI (float, optional): 显著性水平. Defaults to 0.95.

    Returns:
        dataarray: 根据return_mask的值返回相关系数之间的差及其p值或返回mask
    """    
    Z1 = np.arctanh(r1)
    Z2 = np.arctanh(r2)
    z = (Z1-Z2)/np.sqrt(1./(N1-3.)+1./(N2-3.))
    pvalue = 2*(1.-stats.norm.cdf(abs(z)))
    if return_mask:
        return np.where(pvalue<=1.-CI,1.0,0.0)
    else:
        return z,pvalue
    
def cal_mmemask(da, **kargs):
    args = {"dim":"models", "percent": 0.70}
    args = {**args, **kargs}
    mask = xr.where(xr.where(da>0, 1.0, 0.0).mean(dim="models")>=args["percent"], 1.0, 0.0)+xr.where(xr.where(da<0, 1.0, 0.0).mean(dim="models")>=args["percent"], -1.0, 0.0)
    return xr.where(mask*da.mean(dim="models",skipna=True)>0, 1.0, 0.0)


def MME_mask(da, **kargs):
    args = {"dim":"models", "percent":0.70, "chenandyu":False, "big":False}
    args = {**args, **kargs}
    lamb = 1.96
    lim = da.std(dim=args["dim"], skipna=True)*lamb/np.sqrt(len(da.coords[args["dim"]]))
    mask1 = xr.where(abs(da.mean(dim=args["dim"]))-lim>=0.0, 1.0, 0.0)
    mask2 = xr.where((xr.where(xr.where(da>0, 1.0, 0.0).mean(dim="models")>=args["percent"], 1.0, 0.0) + xr.where(xr.where(da<0, 1.0, 0.0).mean(dim="models")>=args["percent"], -1.0, 0.0))*da.mean(dim=args["dim"], skipna=True)>0, 1.0, 0.0)
    if args["chenandyu"] and args["big"]:
        return xr.where(mask1+mask2 >= 2.0, 1.0, 0.0)
    elif args["chenandyu"]:
        return mask1
    elif args["big"]:
        return mask2
    else:
        print("What type of significant test do you want???")

# md:指数计算类
def IWF(u,v):
    """计算IWF指数，其定义为850hPa 5.0°-32.5°N, 90°-140°W区域的涡度
        参考文献：(Wang et al. 2008)(Wang and Fan 1999)
    Args:
        u (dataarray): u风场（多个垂直层次）
        v (dataarray): v风场（多个垂直层次）

    Returns:
        dataarray: IWF index
    """    
    lat = u.coords["lat"]
    lon = v.coords["lon"]
    lat_range = lat[(lat >= 5.0) & (lat <= 32.5)]
    lon_range = lon[(lon >= 90.0) & (lon <= 140.0)]
    u_IWF = u.sel(lat=lat_range, lon=lon_range, level=850.0)
    v_IWF = v.sel(lat=lat_range, lon=lon_range, level=850.0)
    vort = mpcalc.vorticity(u_IWF, v_IWF)
    IWF = cal_area_weighted_mean(vort).metpy.dequantify()
    return IWF    
    del (
        lat,
        lon,
        lat_range,
        lon_range,
        u_IWF,
        v_IWF,
        vort,
        IWF
    )
    
def LKY(u,v):
    """计算LKY指数，其定义为200hPa涡度, (20°–50°N, 110°–150°E)
        参考文献：(Wang et al. 2008)(Lau et al. 2000)

    Args:
        u (dataarray): u风场（多个垂直层次）
        v (dataarray): v风场（多个垂直层次）

    Returns:
        dataarray: LKY index
    """    
    lat = u.coords["lat"]
    lon = v.coords["lon"]
    lat_range = lat[(lat >= 20.0) & (lat <= 50.0)]
    lon_range = lon[(lon >= 110.0) & (lon <= 150.0)]
    u_LKY = u.sel(lat=lat_range, lon=lon_range, level=200.0)
    v_LKY = v.sel(lat=lat_range, lon=lon_range, level=200.0)
    vort = mpcalc.vorticity(u_LKY, v_LKY)
    LKY = cal_lat_weighted_mean(vort).mean(dim="lon", skipna=True).metpy.dequantify()
    return LKY    
    del (
        lat,
        lon,
        lat_range,
        lon_range,
        u_LKY,
        v_LKY,
        vort,
        LKY
    )
    
def NEWI(u):
    """计算NEWI指数，NEWI 5 Nor[u(2.5°-10°N, 105°-140°E) - u(17.5°-22.5°N, 105°-140°E) + u(30°-37.5°N, 105°-140°E)]
        参考文献：(Zhao et al. 2015)

    Args:
        u (dataarray): u风场（多个垂直层次）

    Returns:
        dataarray: NEWI index
    """    
    u200 = u.sel(level=200.0)
    u200_area1 = cal_lat_weighted_mean(u200.loc[:,2.5:10.0,105:140]).mean(dim="lon", skipna=True)
    u200_area2 = cal_lat_weighted_mean(u200.loc[:,17.5:22.5,105:140]).mean(dim="lon", skipna=True)
    u200_area3 = cal_lat_weighted_mean(u200.loc[:,30.0:37.5,105:140]).mean(dim="lon", skipna=True)
    NEWI = standardize(u200_area1 - u200_area2 + u200_area3)
    return NEWI

def SAM(v):
    """计算SAM index，v为monthly meridional wind，并需要包含850hPa和200hPa；本计算已进行纬度加权；
        SAM定义为：10°-30°N，70°-110°E	v850-v200 (Goswami et al. 1999)

    Args:
        v (dataarray): v风场（多个垂直层次）

    Returns:
        dataarray: SAM指数
    """    
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

def SEAM(u):
    """计算SEAM index，u为monthly zonal wind，并需要包含850hPa层次；本计算已进行纬度加权；SEAM定义为：region1 u850 - region2 u850，其中region1: 5°N-15°N，90°E-130°E，region2: 22.5°N-32.5°N，110°E-140°E；参考文献：(Wang and Fan 1999) DU2 index

    Args:
        u (dataarray): u风场（多个垂直层次）

    Returns:
        dataarray: SEAM index
    """    
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

def EAM(u):
    """计算EAM index，其定义为region1: 40°N-50°N，110°E-150°E，region2: 25°N-35°N，110°E-150°E，region1 u200 – region2 u200
        参考文献：(Lau et al. 2000) RM2 index

    Args:
        u (dataarray): u风场（多垂直层次）

    Returns:
        dataarray: EAM index
    """    
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

def WY(u):
    """计算Webster-Yang index；u为zonal wind，需包含850hPa和200hPa，定义为5°N-20°N，40°E-110°E u850-u200

    Args:
        u (dataarray): u风场（多垂直层次）

    Returns:
        dataarray: WY index
    """    
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


# md: 统计类——其他分析方法
def eof_analyse(da,lat,num, north=False,montecarlo=False,montecarlo_times=100,montecarlo_test=0.95):
    """计算da的eof分析，da需要已经标准化处理过、去趋势的变量

    Args:
        da (ndarray): 变量场，需要已经经过标准化和detrend处理
        lat (ndarray): 变量的纬度，用于计算纬度加权
        num (int): 返回num个模态的pc序列
        north (bool, optional):是否使用north检验. Defaults to False.
        montecarlo (bool, optional): 是否应用Monte Carlo检验. Defaults to False.
        monteccarlo_times (int, optional): Monte Carlo重抽样的次数. Defaults to 100.
        montecarlo_test (float, optional): Monte Carlo检验的置信度水平. Defaults to 0.95.

    Returns:
        ndarray: EOF pattern
        ndarray: PC
        ndarray: 模态贡献
    """    
    coslat = np.cos(np.deg2rad(lat))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(da, weights=wgts)
    EOFs = solver.eofs(neofs = num, eofscaling = 2)
    PCs = solver.pcs(npcs = num, pcscaling = 1)
    eigen_Values = solver.eigenvalues()
    percentContrib = eigen_Values * 100./np.sum(eigen_Values)
    north = solver.northTest()
    if montecarlo:
        test = []
        test_new = []
        pass_mode = []
        for i in range(montecarlo_times):
            montecarlo_test = np.random.normal(size=da.shape)
            test_solver = Eof((montecarlo_test-np.mean(montecarlo_test))/np.std(montecarlo_test, axis=0), weights=wgts)
            test_eigen_Values = test_solver.eigenvalues()
            test.append(test_eigen_Values*100./np.sum(test_eigen_Values))
        for j in range(da.shape[0]):
            for i in range(montecarlo_times):
                test_new.append(test[i][j])
                test_new.sort()
            if percentContrib[j]>=test_new[int(montecarlo_times*montecarlo_test)]:
                pass_mode.append("True")
            else:
                pass_mode.append("False")
        print(pass_mode)
        del(i, j, montecarlo_test, test_solver, test_eigen_Values, test, test_new, pass_mode)
        if north:
            print([True if abs(percentContrib[t]-percentContrib[t+1])/north[t]>=1.0 else False for t in range(len(percentContrib)-1)])
            return EOFs,PCs,percentContrib
        else:
            return EOFs,PCs,percentContrib


# md:动力学计算
def cal_ridge_line(da, ridge_trough="min", **kargs):
    """利用u风场计算高压、低压的脊线/槽线，将u的东西风转换位置认为是脊线/槽线
    
    Args:
        da (dataarray): u风场
        ridge_trough (str, optional): Defaults to "min".
        **kargs : 传入lat和lon
    Returns:
        dataarray: 脊线的纬度
        dataarray: 脊线的经度
    """   
    args = dict(lat=da.coords["lat"],lon=da.coords["lon"]) 
    args = {**args, **kargs}
    ridgelon = np.array(args["lon"])
    ridgelat = np.zeros(len(ridgelon))
    for i, longitude in enumerate(ridgelon):
        if ridge_trough == "min":
            ridgelat[i] = float(args["lat"][abs(da.sel(lon=longitude)).argmin(dim="lat")])
        elif ridge_trough == "max":
            ridgelat[i] = float(args["lat"][abs(da.sel(lon=longitude)).argmax(dim="lat")])
    return ridgelat, ridgelon


#   this function is useless because mpcalc.divergence(da1, da2) can calculate the divergence straightforwardly
def cal_divergence(da1, da2):
    """计算散度

    Args:
        da1 (dataarray): u
        da2 (dataarray): v

    Returns:
        dataarray: 散度场
    """    
    dx, dy = mpcalc.lat_lon_grid_deltas(da1.coords["lon"], da1.coords["lat"])
    div = xr.apply_ufunc(
        mpcalc.divergence,
        da1,
        da2,
        input_core_dims=[["lat","lon"],["lat","lon"]],
        output_core_dims=[["lat","lon"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"dx":dx, "dy":dy}
    )
    return div

# md:其他
def retrieve_allstrindex(filename, strkey):
    """返回字符串中，需要搜索的字段的所在位置

    Args:
        filename (str): 字符串
        strkey (str): 需要搜索的字段

    Returns:
        int: 位置
    """    
    count = filename.count(strkey)
    c = -1
    loc = []
    for i in range(count):
        c = filename.find(strkey, c+1, len(filename))
        loc.append(c)
    return loc



# %%