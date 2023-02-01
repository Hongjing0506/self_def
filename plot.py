'''
Author: ChenHJ
Date: 2021-12-12 21:40:15
LastEditors: ChenHJ
LastEditTime: 2022-12-06 12:02:04
FilePath: /chenhj/self_def/plot.py
Aim: 
Mission: 
'''
import numpy as np
import xarray as xr
import os
import re
from cdo import *
import shutil
import pandas as pd

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
return {*}
log:
updated in 2022/07/08: 
    add **kargs, including: 
		majorticklabelsize: the font size of major tick labels
		majorticklen: the length of major tick
		minorticklen: the length of minor tick
		coastcolor: the color of coastline
		coastlinewidth: the line width of coastline
		lonminorspace: the space of minor longitude ticks
		latminorspace: the space of minor latitude ticks
		majorticklabelpad: the pad between tick and ticklabel
    fixing bugs:
		fixed the bug that when central_longitude was not 0, set_extent attribute became invalid
'''    
def geo_ticks(axs, lonticks, latticks, cl, extents, **kargs):
    args = {"majorticklabelsize":7, "majorticklen":4.0, "minorticklen":3.0, "coastcolor":"grey6", "coastlinewidth":1.3, "lonminorspace":10, "latminorspace":5, "majorticklabelpad":2.0}
    args = {**args, **kargs}
    import proplot as pplt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter
    from cartopy.mpl.ticker import LatitudeFormatter
    from matplotlib.ticker import MultipleLocator
    axs.format(coast=True, coastlinewidth=args["coastlinewidth"], coastzorder=1, coastcolor=args["coastcolor"], labels=False, grid=False)
    proj = ccrs.PlateCarree(central_longitude=cl)
    lonticks += -cl
    extents[0] = extents[0]-cl
    extents[1] = extents[1]-cl
    axs.set_extent(extents, crs=proj)
    axs.set_xticks(lonticks, crs=proj)
    axs.set_yticks(latticks, crs=proj)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    axs.minorticks_on()
    xminorLocator = MultipleLocator(args["lonminorspace"])
    yminorLocator = MultipleLocator(args["latminorspace"])
    for ax in axs:
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.xaxis.set_minor_locator(xminorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=args["majorticklabelsize"],
            direction="out",
            length=args["majorticklen"],
            width=0.8,
            pad=args["majorticklabelpad"],
            top=False,
            right=False,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            length=args["minorticklen"],
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

def plt_sig(da, ax, n, area, color, large, marker=".", **kargs):
    """对需要的位置打点

    Args:
        da (_type_): _description_
        ax (_type_): _description_
        n (_type_): _description_
        area (_type_): _description_
        color (_type_): _description_
        large (_type_): _description_
        marker (str, optional): _description_. Defaults to ".".
    """
    da_cyc, lon_cyc = add_cyclic_point(da[::n, ::n], coord=da.lon[::n])
    nx, ny = np.meshgrid(lon_cyc, da.lat[::n])
    sig = ax.scatter(
        nx[area], ny[area], marker=marker, s=large, c=color, alpha=0.6, **kargs)

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
    
def patches(ax, x0, y0, width, height, proj, **kargs):
    from matplotlib.patches import Rectangle
    args = {"edgecolor":"grey7", "linewidth": 1.0, "linestyle": "--"}
    args = {**args, **kargs}
    rect = Rectangle(
        (x0, y0),
        width,
        height,
        fc="none",
        ec=args['edgecolor'],
        linewidth=args['linewidth'],
        zorder=1.1,
        transform=proj,
        linestyle=args["linestyle"],
    )
    ax.add_patch(rect)
    
def taylor_diagram(ax,r,std,**kargs):
    dotlables=["" for _ in range(len(r))]
    args = {"dotlables":dotlables, "color":"r", "lables":False}
    args = {**args, **kargs}
    if np.min(r)<0:
        ax.set_thetalim(thetamin=0, thetamax=180)
        r_small, r_big, r_interval=0,1.5+0.1,0.5  #横纵坐标范围，最小值 最大值 间隔
        ax.set_rlim(r_small,r_big)
        rad_list=[-1,-0.99,-0.95,-0.9,-0.8,-0.7,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.7,0.8,0.9,0.95,0.99,1] #需要显示数值的主要R的值
        # minor_rad_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.86,0.87,0.88,0.89,
                        # 0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1] #需要显示刻度的次要R的值
        angle_list = np.rad2deg(np.arccos(rad_list))
        angle_list_rad=np.arccos(rad_list)
        # angle_minor_list = np.arccos(minor_rad_list)
        ax.set_thetagrids(angle_list, rad_list)
        ax.tick_params(pad=8)
        # lines, labels = plt.thetagrids(angle_list, labels=rad_list, frac=1.25)
        v = 0.11
        for i in np.arange(r_small, r_big, r_interval):
            if i == 1:
                ax.text(np.arctan(i/v)-0.5*np.pi, np.sqrt(v**2+i**2), s='REF', ha='center', va='top') 
                ax.text(1.5*np.pi-np.arctan(i/v), np.sqrt(v**2+i**2), s='REF', ha='center', va='top')
                #text的第一个坐标是角度（弧度制），第二个是距离
            elif i == 0:
                ax.text(1.5*np.pi, v, s=str(i), ha='center', va='top')
            else: 
                ax.text(np.arctan(i/v)-0.5*np.pi, np.sqrt(v**2+i**2), s=str(i), ha='center', va='top') 
                ax.text(1.5*np.pi-np.arctan(i/v), np.sqrt(v**2+i**2), s=str(i), ha='center', va='top') 
        ax.set_rgrids([])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        ax.grid(False)
        angle_linewidth,angle_length,angle_minor_length=0.8,0.02,0.01
        tick = [ax.get_rmax(), ax.get_rmax() * (1 - angle_length)]
        # tick_minor = [ax.get_rmax(), ax.get_rmax() * (1 - angle_minor_length)]
        for t in angle_list_rad:
            ax.plot([t, t], tick, lw=angle_linewidth, color="k")  # 第一个坐标是角度（角度制），第二个是距离
        # for t in angle_minor_list:
        #     ax.plot([t, t], tick_minor, lw=angle_linewidth, color="k")  # 第一个坐标是角度（角度制），第二个是距离

        # 然后开始绘制以REF为原点的圈，可以自己加圈
        circle = plt.Circle((1, 0), 0.5, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='gray',linestyle='--', linewidth=0.8)
        ax.add_artist(circle)
        circle = plt.Circle((1, 0), 1.0, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='gray',linestyle='--', linewidth=0.8)
        ax.add_artist(circle)

        # 绘制以原点为圆点的圆圈：
        circle4 = plt.Circle((0, 0), 0.5, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',linestyle='--', linewidth=1.0)
        circle5 = plt.Circle((0, 0), 1, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',linestyle='-', linewidth=1.5)
        circle6 = plt.Circle((0, 0), 1.5, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',linestyle='--', linewidth=1.0)
        ax.add_artist(circle4)
        ax.add_artist(circle5)
        ax.add_artist(circle6)

        #ax.set_xlabel('Normalized')
        ax.text(np.deg2rad(40), 1.85, s='Correlation', ha='center', va='bottom', rotation=-45)  

        # 这里的网格包括：以原点为圆点的圆圈。首先绘制从原点发散的线段，长度等于半径
        ax.plot([0,np.arccos(0.4)],[0,3],lw=1,color='gray',linestyle='--')
        ax.plot([0,np.arccos(0.8)],[0,3],lw=1,color='gray',linestyle='--')
        ax.plot([0,np.arccos(0.0)],[0,3],lw=1,color='gray',linestyle='--')
        ax.plot([0,np.arccos(-0.4)],[0,3],lw=1,color='gray',linestyle='--')
        ax.plot([0,np.arccos(-0.8)],[0,3],lw=1,color='gray',linestyle='--')

        # 画点，参数一相关系数，参数二标准差 
        for i in np.arange(0,len(r)):
            if args["lables"]:
                ax.plot(np.arccos(r[i]), std[i], 'o',color=args["color"],markersize=0, label='{} {}'.format(i+1,args["dotlables"][i]))
            else:
                ax.plot(np.arccos(r[i]), std[i], 'o',color=args["color"],markersize=0)
            ax.text(np.arccos(r[i]), std[i], s='{}'.format(i+1), c=args["color"],fontsize=10)
            # ax.text(np.arccos(r[i]-0.05), std[i], s='2', c='r',fontsize=13)
            # ax.plot(np.arccos(r[0]), std[0], 'o',color='#FF8000',markersize=10, label='1')
            # ax.text(np.arccos(r[0]-0.05), std[0], s='1', c='#FF8000',fontsize=13)
        # ax.plot(np.arccos(r[2]), std[2], 'o',color='g',markersize=10, label='3')
        # ax.text(np.arccos(r[2]-0.05), std[2], s='3',c='g', fontsize=13)
        # ax.plot(np.arccos(r[3]), std[3], 'o',color='b',markersize=10, label='4')
        # ax.text(np.arccos(r[3]-0.05), std[3], s='4', c='b',fontsize=13)
        # ax.plot(np.arccos(r[4]), std[4], 'o',color='#E800E8',markersize=10, label='5')
        # ax.text(np.arccos(r[4]-0.05), std[4], s='5', c='#E800E8',fontsize=13)
        # ax.plot(np.arccos(r[5]), std[5], '^',color='#00AEAE',markersize=10, label='6')
        # ax.text(np.arccos(r[5]-0.05), std[5], s='6', c='#00AEAE',fontsize=13)
        ax.text(1.5*np.pi, 0.3, s='Std (Normalized)',ha='center', va='top')
    elif np.min(r)>=0:
        ax.set_thetalim(thetamin=0, thetamax=90)
        r_small, r_big, r_interval=0,1.5+0.1,0.5  #横纵坐标范围，最小值 最大值 间隔
        ax.set_rlim(r_small,r_big)
        rad_list=[0,0.2,0.4,0.6,0.7,0.8,0.9,0.95,0.99,1] #需要显示数值的主要R的值
        # minor_rad_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.86,0.87,0.88,0.89,
                        # 0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1] #需要显示刻度的次要R的值
        angle_list = np.rad2deg(np.arccos(rad_list))
        angle_list_rad=np.arccos(rad_list)
        # angle_minor_list = np.arccos(minor_rad_list)
        ax.set_thetagrids(angle_list, rad_list)
        
        #   设置pcc和圆弧的距离
        ax.tick_params(pad=8)
        # lines, labels = plt.thetagrids(angle_list, labels=rad_list, frac=1.25)
        #   画出Std的坐标轴（横向）
        v = 0.07
        for i in np.arange(r_small, r_big, r_interval):
            if i == 1:
                ax.text(np.arctan(i/v)-0.5*np.pi, np.sqrt(v**2+i**2), s='REF', ha='center', va='top') 
                ax.text(0.5*np.pi+np.arctan(v/i), np.sqrt(v**2+i**2), s='REF', ha='center', va='top')
                #text的第一个坐标是角度（弧度制），第二个是距离
            elif i == 0:
                ax.text(1.5*np.pi, v, s=str(i), ha='center', va='top')
                ax.text(np.pi, v, s=str(i), ha='center', va='top')
            else: 
                ax.text(np.arctan(i/v)-0.5*np.pi, np.sqrt(v**2+i**2), s=str(i), ha='center', va='top') 
                ax.text(0.5*np.pi+np.arctan(v/i), np.sqrt(v**2+i**2), s=str(i), ha='center', va='top')
                # ax.text(1.5*np.pi-np.arctan(i/v), np.sqrt(v**2+i**2), s=str(i), ha='center', va='top') 
        ax.set_rgrids([])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        ax.grid(False)
        angle_linewidth,angle_length,angle_minor_length=0.8,0.02,0.01
        tick = [ax.get_rmax(), ax.get_rmax() * (1 - angle_length)]
        # tick_minor = [ax.get_rmax(), ax.get_rmax() * (1 - angle_minor_length)]
        for t in angle_list_rad:
            ax.plot([t, t], tick, lw=angle_linewidth, color="k")  # 第一个坐标是角度（角度制），第二个是距离
        # for t in angle_minor_list:
        #     ax.plot([t, t], tick_minor, lw=angle_linewidth, color="k")  # 第一个坐标是角度（角度制），第二个是距离

        # 然后开始绘制以REF为原点的圈，可以自己加圈
        circle = plt.Circle((1, 0), 0.5, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='gray',linestyle='--', linewidth=0.8)
        ax.add_artist(circle)
        circle = plt.Circle((1, 0), 1.0, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='gray',linestyle='--', linewidth=0.8)
        ax.add_artist(circle)

        # 绘制以原点为圆点的圆圈：
        circle4 = plt.Circle((0, 0), 0.5, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',linestyle='--', linewidth=1.0)
        circle5 = plt.Circle((0, 0), 1, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',linestyle='-', linewidth=1.5)
        circle6 = plt.Circle((0, 0), 1.5, transform=ax.transData._b, facecolor=(0, 0, 0, 0), edgecolor='grey',linestyle='--', linewidth=1.0)
        ax.add_artist(circle4)
        ax.add_artist(circle5)
        ax.add_artist(circle6)

        #ax.set_xlabel('Normalized')
        ax.text(np.deg2rad(40), 1.85, s='Correlation', ha='center', va='bottom', rotation=-45)  

        # 这里的网格包括：以原点为圆点的圆圈。首先绘制从原点发散的线段，长度等于半径
        ax.plot([0,np.arccos(0.4)],[0,3],lw=1,color='gray',linestyle='--')
        ax.plot([0,np.arccos(0.8)],[0,3],lw=1,color='gray',linestyle='--')
        

        # 画点，参数一相关系数，参数二标准差 
        for i in np.arange(0,len(r)):
            if args["lables"]:
                ax.plot(np.arccos(r[i]), std[i], 'o',color=args["color"],markersize=0, label='{} {}'.format(i+1,args["dotlables"][i]))
            else:
                ax.plot(np.arccos(r[i]), std[i], 'o',color=args["color"],markersize=0)
            ax.text(np.arccos(r[i]), std[i], s='{}'.format(i+1), c=args["color"],fontsize=10)
            # ax.text(np.arccos(r[i]-0.05), std[i], s='2', c='r',fontsize=13)
            # ax.plot(np.arccos(r[0]), std[0], 'o',color='#FF8000',markersize=10, label='1')
            # ax.text(np.arccos(r[0]-0.05), std[0], s='1', c='#FF8000',fontsize=13)
        # ax.plot(np.arccos(r[2]), std[2], 'o',color='g',markersize=10, label='3')
        # ax.text(np.arccos(r[2]-0.05), std[2], s='3',c='g', fontsize=13)
        # ax.plot(np.arccos(r[3]), std[3], 'o',color='b',markersize=10, label='4')
        # ax.text(np.arccos(r[3]-0.05), std[3], s='4', c='b',fontsize=13)
        # ax.plot(np.arccos(r[4]), std[4], 'o',color='#E800E8',markersize=10, label='5')
        # ax.text(np.arccos(r[4]-0.05), std[4], s='5', c='#E800E8',fontsize=13)
        # ax.plot(np.arccos(r[5]), std[5], '^',color='#00AEAE',markersize=10, label='6')
        # ax.text(np.arccos(r[5]-0.05), std[5], s='6', c='#00AEAE',fontsize=13)
        ax.set_ylabel('Std (Normalized)',labelpad=35)


def contour_label(ax, contourf_instance, **kargs):
  xmin,xmax,ymin,ymax = plt.axis()
  plotmid = (xmin+xmax)/2, (ymin+ymax)/2
  label_pos = []
  for num_line,line in enumerate(contourf_instance.collections):
    for path in line.get_paths():
      # 获取contour线的每一点的坐标
      vert = path.vertices

      # 寻找最接近中心的点
      dist = np.linalg.norm(vert-plotmid, ord=2, axis=1)
      min_ind = np.argmin(dist)
      label_pos.append(vert[min_ind-num_line+len(contourf_instance.collections)//2,:])
  ax.clabel(contourf_instance, manual=label_pos, **kargs)