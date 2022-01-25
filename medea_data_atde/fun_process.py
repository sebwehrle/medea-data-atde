# %% imports
import pandas as pd
import os
import numpy as np
from scipy import interpolate
from datetime import datetime as dt
from netCDF4 import Dataset, num2date
from medea_data_atde.fun_get import days_in_year, resample_index


# %% functions for heat load calculation
# ----------------------------------------------------------------------------------------------------------------------
def heat_yr2day(av_temp, ht_cons_annual):
    """
    Converts annual heat consumption to daily heat consumption, based on daily mean temperatures.
    Underlying algorithm relies on https://www.agcs.at/agcs/clearing/lastprofile/lp_studie2008.pdf
    Implemented consumer clusters are for residential and commercial consumers:
    * HE08 Heizgas Einfamilienhaus LP2008
    * MH08 Heizgas Mehrfamilienhaus LP2008
    * HG08 Heizgas Gewerbe LP2008
    Industry load profiles are specific and typically measured, i.e. not approximated by load profiles

    :param av_temp: datetime-indexed pandas.DataFrame holding daily average temperatures
    :param ht_cons_annual:
    :return:
    """
    # ----------------------------------------------------------------------------
    # fixed parameter: SIGMOID PARAMETERS
    # ----------------------------------------------------------------------------
    sigm_a = {'HE08': 2.8423015098, 'HM08': 2.3994211316, 'HG08': 3.0404658371}
    sigm_b = {'HE08': -36.9902101066, 'HM08': -34.1350545407, 'HG08': -35.6696458089}
    sigm_c = {'HE08': 6.5692076687, 'HM08': 5.6347421440, 'HG08': 5.6585923962}
    sigm_d = {'HE08': 0.1225658254, 'HM08': 0.1728484079, 'HG08': 0.1187586955}

    # ----------------------------------------------------------------------------
    # breakdown of annual consumption to daily consumption
    # ----------------------------------------------------------------------------
    # temperature smoothing
    temp_smooth = pd.DataFrame(index=av_temp.index, columns=['Temp_Sm'])
    for d in av_temp.index:
        if d >= av_temp.first_valid_index() + pd.Timedelta(1, unit='d'):
            temp_smooth.loc[d] = 0.5 * av_temp.loc[d] \
                                 + 0.5 * temp_smooth.loc[d - pd.Timedelta(1, unit='d')]
        else:
            temp_smooth.loc[d] = av_temp.loc[d]

    # determination of normalized daily consumption h_value
    h_value = pd.DataFrame(index=av_temp.index, columns=sigm_a.keys())
    for key in sigm_a:
        h_value[key] = sigm_a[key] / (1 + (sigm_b[key] / (temp_smooth.values - 40)) ** sigm_c[key]) + sigm_d[key]

    # generate matrix of hourly annual consumption
    annual_hourly_consumption = pd.DataFrame(1, index=av_temp.index, columns=sigm_a.keys())
    annual_hourly_consumption = annual_hourly_consumption.set_index(annual_hourly_consumption.index.year, append=True)
    annual_hourly_consumption = annual_hourly_consumption.multiply(ht_cons_annual, level=1).reset_index(drop=True,
                                                                                                        level=1)

    # generate matrix of annual h-value sums
    h_value_annual = h_value.groupby(h_value.index.year).sum()

    # de-normalization of h_value
    cons_daily = h_value.multiply(annual_hourly_consumption)
    cons_daily = cons_daily.set_index(cons_daily.index.year, append=True)
    cons_daily = cons_daily.divide(h_value_annual, level=1).reset_index(drop=True, level=1)

    return cons_daily


def heat_day2hr(df_ht, con_day, con_pattern):
    """
    convert daily heat consumption to hourly heat consumption
    Underlying algorithm relies on https://www.agcs.at/agcs/clearing/lastprofile/lp_studie2008.pdf
    ATTENTION: Algorithm fails for daily average temperatures below -25°C !
    :param df_ht:
    :param con_day:
    :param con_pattern:
    :return:
    """
    sigm_a = {'HE08': 2.8423015098, 'HM08': 2.3994211316, 'HG08': 3.0404658371}
    # apply demand_pattern
    last_day = pd.DataFrame(index=df_ht.tail(1).index + pd.Timedelta(1, unit='d'), columns=sigm_a.keys())

    cons_hourly = con_day.append(last_day).astype(float).resample('1H').sum()
    cons_hourly.drop(cons_hourly.tail(1).index, inplace=True)

    for d in df_ht.index:
        temp_lvl = np.floor(df_ht[d] / 5) * 5
        cons_hlpr = con_day.loc[d] * con_pattern.loc[temp_lvl]
        cons_hlpr = cons_hlpr[cons_hourly.columns]
        cons_hlpr.index = d + pd.to_timedelta(cons_hlpr.index, unit='h')
        cons_hourly.loc[cons_hlpr.index] = cons_hlpr.astype(str).astype(float)

    cons_hourly = cons_hourly.astype(str).astype(float)
    return cons_hourly


def mean_temp_at_plants(db_plants, era_dir, country, years, zones):
    temp_date_range = pd.date_range(pd.datetime(years[0], 1, 1), pd.datetime(years[-1], 12, 31), freq='D')
    daily_mean_temp = pd.DataFrame(index=temp_date_range.values, columns=[zones])
    for zne in zones:
        chp = db_plants[(db_plants['UnitCoGen'] == 1) & (db_plants['UnitNameplate'] >= 10) &
                        (db_plants['PlantCountry'] == country[zne])]
        chp_lon = chp['PlantLongitude'].values
        chp_lat = chp['PlantLatitude'].values
        for year in years:
            filename = os.path.join(era_dir, f'temperature_europe_{year}.nc')
            era5 = Dataset(filename, format='NETCDF4')
            # get grid
            lats = era5.variables['latitude'][:]  # y
            lons = era5.variables['longitude'][:]  # x
            DAYS = range(0, days_in_year(year), 1)
            for day in DAYS:
                hour = day * 24
                temp_2m = era5.variables['t2m'][hour, :, :] - 273.15
                # obtain weighted average temperature from interpolation function, using CHP capacities as weights
                f = interpolate.interp2d(lons, lats, temp_2m)
                temp_itp = np.diagonal(f(chp_lon, chp_lat)) * chp['UnitNameplate'].values / chp['UnitNameplate'].sum()
                era_date = num2date(era5.variables['time'][hour], era5.variables['time'].units,
                                    era5.variables['time'].calendar)
                # conversion of date formats
                era_date = dt.fromisoformat(era_date.isoformat())
                daily_mean_temp.loc[era_date, zne] = np.nansum(temp_itp)
                # print(era_date)
    # export results
    daily_mean_temp.replace(-9999, np.nan, inplace=True)
    return daily_mean_temp


def heat_consumption(zones, years, cons_annual, df_heat, cons_pattern):
    # ----------------------------------------------------------------------------
    dayrange = pd.date_range(pd.datetime(np.min(df_heat['year']), 1, 1), pd.datetime(np.max(df_heat['year']), 12, 31),
                             freq='D')

    # calculate heat consumption for each region
    # ----------------------------------------------------------------------------
    idx = pd.IndexSlice
    regions = zones
    ht_consumption = pd.DataFrame(index=resample_index(df_heat.index, 'h'), columns=cons_annual.columns)
    for reg in regions:
        cons_daily = heat_yr2day(df_heat[reg], cons_annual.loc[:, reg])
        cons_hourly = heat_day2hr(df_heat[reg], cons_daily, cons_pattern)
        cons_hourly.columns = pd.MultiIndex.from_product([[reg], cons_hourly.columns])
        ht_consumption.loc[:, idx[reg, :]] = cons_hourly

    # fill WW and IND
    for yr in years:
        for zn in zones:
            hrs = len(ht_consumption.loc[str(yr), :])
            ht_consumption.loc[str(yr), idx[zn, 'WW']] = cons_annual.loc[yr, idx[zn, 'WW']] / hrs
            ht_consumption.loc[str(yr), idx[zn, 'IND']] = cons_annual.loc[yr, idx[zn, 'IND']] / hrs

    return ht_consumption
