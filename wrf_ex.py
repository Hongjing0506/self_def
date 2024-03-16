'''
Author: ChenHJ
Date: 2024-03-16 15:50:10
LastEditors: ChenHJ
LastEditTime: 2024-03-16 16:16:08
FilePath: /chenhj/self_def/wrf_ex.py
Aim: 
Mission: 
'''
#%%
import xarray as xr
import numpy as np
import wrf
import sys
sys.path.append("/home/ys17-23/chenhj/self_def")
import cal as ca
import os
import pandas as pd

#%%
# md: 处理静态数据
def generate_static_data(static_path, wpsoutput_path,static_info_path):
  """检查是否存在静态数据及网格坐标信息。如果不存在，则生成。

  Args:
      static_path (str): 静态数据（含LANDMASK）文件路径
      wpsoutput_path (str): geo_em文件所在文件夹的路径
      static_info_path (str): 网格坐标信息文件路径
  """ 
  if not os.path.exists(static_path):
    # 静态数据不存在，生成静态数据及网格坐标信息
    wrfstatic = xr.open_dataset(os.path.join(wpsoutput_path,"geo_em.d01.nc"))
    xlat = wrfstatic["XLAT_M"]
    xlong = wrfstatic["XLONG_M"]
    landmask = wrfstatic["LANDMASK"]
    # landuse = wrfstatic["LANDUSEF"] #这里的21类是怎么处理的？

    static = xr.Dataset(
      data_vars=dict(
        LANDMASK=(["time","south_north","west_east"],landmask.data),
        # LANDUSE=(["time","south_north","west_east"],landuse.data),
      ),
      coords=dict(
        time=landmask.coords["Time"].data,
        XLAT=(["south_north","west_east"],xlat[0].data),
        XLONG=(["south_north","west_east"],xlong[0].data)
      ),
    )
    static = static.assign_attrs(wrfstatic.attrs)

    for key,values in xlat.attrs.items():
      static.coords["XLAT"].attrs[key]=values
    for key,values in xlong.attrs.items():
      static.coords["XLONG"].attrs[key]=values

    static.coords["XLAT"].attrs["units"] = "degree_north"
    static.coords["XLONG"].attrs["units"] = "degree_east"

    ca.save(static,static_path)
    
    os.system("cdo griddes {} > {}".format(static_path,static_info_path))
    
  else:
    print("Static file has existed!")
    if not os.path.exists(static_info_path):
      # static info不存在，则生成
      os.system("cdo griddes {} > {}".format(static_path,static_info_path))
      print("Successfully generating grid information file in {}".format(static_info_path))
    else:
      print("Grid information file has existed!")

# md: 变量后处理
def _get_pravg(wrfin,var,**kwargs):
  """从wrfout计算总降水率PRAVG

  Args:
      wrfin (dataarray or iterable): 读取的wrfout文件
      var (str): 占位符
  """  
  exvar = wrf.extract_vars(wrfin, varnames=["RAINC", "RAINNC", "RAINSH"],timeidx=wrf.ALL_TIMES)
  if len(exvar["RAINNC"].coords["Time"]) != 1:
    time_int = ((exvar["RAINNC"].coords["Time"][1]-exvar["RAINNC"].coords["Time"][0])/np.timedelta64(1, 's')).data  # time interval
  elif 'time_int' in kwargs:
    time_int = kwargs['time_int']
  else:
    time_int = 1.
  
  # 处理累积量变为降水率
  rainnc = exvar["RAINNC"]
  rainnc_new = rainnc.diff(dim="Time",n=1)/time_int
  
  rainc = exvar["RAINC"]
  rainc_new = rainc.diff(dim="Time",n=1)/time_int
  
  rainsh = exvar["RAINSH"]
  rainsh_new = rainsh.diff(dim="Time",n=1)/time_int
  
  rain = rainnc+rainc+rainsh
  rain.name="PRAVG"
  rain.attrs["units"]="mm s-1"
  
  return(rain)

def _get_freq(var, freq, style="mean"):
  """根据指定的时间频率和方式进行时间重采样

  Args:
      var (dataarray): 数据
      freq (str): 时间频率，daily or 3hourly或直接指定
      style (str, optional): 重采样方式，mean, min or max. Defaults to "mean".
  """  
  if freq == "daily":
    fre = "D"
  elif freq == "3hourly":
    fre = "3H"
  else:
    fre = freq
  if style=="mean":
    var_daily = var.resample(Time=fre).mean()
  elif style=="min":
    var_daily = var.resample(Time=fre).min()
  elif style=="max":
    var_daily = var.resample(Time=fre).max()
  # # modify the var's attibutes
  # var_daily.name=varname
  # var_daily = var_daily.rename({"dayofyear":"time"})
  # var_daily.attrs["projection"]=""
  # time=var.resample(Time='D').first()['Time']
  # time = pd.to_datetime(time.values)
  # time = pd.date_range(start=time[0].normalize(), periods=len(time), freq='D')
  # var_daily.coords["time"] = time
  return(var_daily)


def _vinterp(wrfin,var,varkey='pres',lev=[1000,975,950,925,900,850,800,700,600,500,400,300,250,200,150,100,70]):
  """垂直插值函数

  Args:
      wrfin (dataarray or iterable): 读取的wrfout文件
      var (dataarray): 数据
      varkey (str, optional): 垂直插值的坐标系. Defaults to 'pres'.
      lev (list, optional): 垂直插值层次. Defaults to [1000,975,950,925,900,850,800,700,600,500,400,300,250,200,150,100,70].

  Returns:
      dataarray: 插值好的数据
  """  
  # 计算垂直插值至17层
  ans = wrf.vinterp(wrfin,var,varkey,lev,timeidx=wrf.ALL_TIMES)
  ans = ans.rename({"interp_level":"lev"})
  return (ans)

def _get_uv10met(wrfin):
  """获取uv10风场结果

  Args:
      wrfin (dataarray or iterable): 读取的wrfout文件
  """  
  uvmet10 = wrf.getvar(wrfin,"uvmet10",units="m s-1",timeidx=wrf.ALL_TIMES)
  u10, v10 = uvmet10[0], uvmet10[1]
  u10 = u10.reset_coords(names=["u_v"],drop=True)
  v10 = v10.reset_coords(names=["u_v"],drop=True)
  u10.name="U10"
  v10.name="V10"
  return(u10,v10)

def _reset_coords_attrs(var):
  """修正变量的坐标名和属性

  Args:
      var (dataarray): 数据
  """  
  cnlist = [cn for cn in var.coords.keys()]
  if 'XTIME' in cnlist:
    var = var.reset_coords(names=["XTIME"],drop=True)
  if 'datetime' in cnlist:
    var = var.reset_coords(names=["datetime"],drop=True)
  if 'Time' in cnlist:
    var = var.rename({'Time':'time'})
  if 'XLAT' in cnlist:
    var = var.reset_coords(names=["XLAT"],drop=True)
  if 'XLONG' in cnlist:
    var = var.reset_coords(names=["XLONG"],drop=True)
  
  attlist = [an for an in var.attrs.keys()]
  if 'projection' in attlist:
    var.attrs.pop("projection",None)
  if 'coordinates' in attlist:
    var.attrs.pop("coordinates",None)
  return(var)

def _wrf_getvar(wrfin,var):
  """从wrfout文件中获取指定变量并按照预定义变换单位和相关属性、描述，处理fillvalue

  Args:
      wrfin (dataarray or iterable): 读取的wrfout文件
      var (str): 数据名
  """  
  var_alias = {
    "H": {"varname":"geopt", "scale_factor":[np.divide,9.81], "new_attrs":{"units":"m", "description":"geopotential heihgt"}},
    "T2MAX": {"varname":"T2M_MAX", "fillvalue":999.},
    "T2MIN": {"varname":"T2M_MIN", "fillvalue":999.},
    "T2M": {"varname":"T2"}
  }
  result = wrf.getvar(wrfin,var_alias[var]["varname"],timeidx=wrf.ALL_TIMES)
  result.name=var
  varkeys = [vk for vk in var_alias[var]]
  if 'scale_factor' in varkeys:
    result = var_alias[var]['scale_factor'][0](result,var_alias[var]['scale_factor'][1])
  if 'new_attrs' in varkeys:
    for nk,nv in var_alias[var]['new_attrs'].items():
      result.attrs[nk] = nv
  if 'fillvalue' in varkeys:
    result = xr.where(result>=var_alias[var]["fillvalue"]-1,np.nan,result)
  return(result)

  
def get_var(wrfin, var, freq,**kwargs):
  """封装函数，从wrfin中提取指定变量，并垂直插值、按照指定的时间频率进行时间重采样、整理坐标维度名和属性等。

  Args:
      wrfin (_type_): _description_
      var (_type_): _description_
      freq (_type_): _description_

  Raises:
      ValueError: _description_
  """  
  _FUNC_MAP = {"U10":[_get_uv10met,0],
              "V10":[_get_uv10met,1],
              "PRAVG":_get_pravg,
              "H":_wrf_getvar,
              "T2MAX":_wrf_getvar,
              "T2MIN":_wrf_getvar,
              "T2M":_wrf_getvar,
  }
  _vinterp_map = ["H"]
  freq_style_map = {
                    "T2MAX": "max",
                    "T2MIN": "min",
                    }
                    
  # 提取变量（含垂直插值）
  func = _FUNC_MAP.get(var)
  if isinstance(func, list):
    result = func[0](wrfin)[func[1]]
  elif callable(func):
    ca._check_kargs(var,kwargs)
    result = func(wrfin,var,**kwargs)
  else:
    raise ValueError(f"No function defined for variable {var}")
  # 垂直插值
  if var in _vinterp_map:
    result = _vinterp(wrfin,result)
  # 时间频次整理
  style="mean"
  if var in freq_style_map.keys():
    style=freq_style_map[var]
  result = _get_freq(result,freq,style=style)
  # 整理坐标维度名和属性
  result = _reset_coords_attrs(result)
  return(result)