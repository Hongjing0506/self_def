'''
Author: ChenHJ
Date: 2021-12-13 21:04:08
LastEditors: ChenHJ
LastEditTime: 2024-01-02 17:53:48
FilePath: /ys17-23/ERA5_download.py
Aim: 
Mission: 
'''
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'v_component_of_wind',
        'pressure_level': ['1000', '850', '700', '500', '200'],
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
        'format': 'netcdf',
        'grid'      :[0.5, 0.5]
    },
    'v_mon_0.5x0.5_1979_2021.nc')