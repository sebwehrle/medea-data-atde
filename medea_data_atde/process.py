# %% imports
import os
import sysconfig
from pathlib import Path
import logging
from medea_data_atde.logging_config import setup_logging
import pandas as pd
import numpy as np
from scipy import interpolate
from datetime import datetime as dt
from netCDF4 import Dataset, num2date
from medea_data_atde.retrieve import days_in_year, resample_index


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
        temp_lvl = np.floor(df_ht.loc[d] / 5) * 5
        cons_hlpr = con_day.loc[d] * con_pattern.loc[temp_lvl]
        cons_hlpr = cons_hlpr[cons_hourly.columns]
        cons_hlpr.index = d + pd.to_timedelta(cons_hlpr.index.astype('int'), unit='h')
        cons_hourly.loc[cons_hlpr.index] = cons_hlpr.astype(str).astype(float)

    cons_hourly = cons_hourly.astype(str).astype(float)
    return cons_hourly


def mean_temp_at_plants(db_plants, era_dir, years, zones):
    temp_date_range = pd.date_range(f'{years[0]}/01/01', f'{years[-1]}/12/31', freq='D')
    daily_mean_temp = pd.DataFrame(index=temp_date_range.values, columns=[zones])
    for zne in zones:
        chp = db_plants[(db_plants['country'] == zne) & (db_plants['chp'] == 'yes')]
        if chp.empty:
            chp = db_plants[(db_plants['country'] == zne) & (db_plants['technology'] == 'Combined cycle')]
        chp_lon = chp['lon'].values
        chp_lat = chp['lat'].values
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
                temp_itp = np.diagonal(f(chp_lon, chp_lat)) * chp['capacity'].values / chp['capacity'].sum()
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
    dayrange = pd.date_range(f'{np.min(df_heat["year"])}/01/01', f'{np.max(df_heat["year"])}/12/31', freq='D')

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


def compile_hydro_generation(root_dir, zones):
    """
    process hydro power generation time series from entso-e transparency platform
    :param root_dir: Path() of project root directory
    :param zones: list of model zones
    :return:
    """
    dir_aggenpertype = root_dir / 'data' / 'raw' / 'AggregatedGenerationPerType_16.1.B_C'

    df_ror = pd.DataFrame()
    for file in os.listdir(dir_aggenpertype):
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith('.csv'):
            df_tmpfile = pd.DataFrame()
            ts_agpt = pd.read_csv(dir_aggenpertype / filename, encoding='utf-8', sep='\t')
            ts_agpt['DateTime'] = pd.to_datetime(ts_agpt['DateTime'])
            ts_agpt.set_index('DateTime', inplace=True)
            # splitstr = filename.split('_')
            # time = pd.datetime(int(splitstr[0]), int(splitstr[1]), 1)

            if any('ProductionType_Name' in s for s in ts_agpt.columns):
                newcols = [col.replace('ProductionType_Name', 'ProductionType') for col in ts_agpt.columns]
                ts_agpt.columns = newcols

            for reg in zones:
                df_tmpror = ts_agpt.loc[(ts_agpt['ProductionType'] == 'Hydro Run-of-river and poundage') & (
                        ts_agpt['MapCode'] == reg), 'ActualGenerationOutput']
                df_tmpres = ts_agpt.loc[(ts_agpt['ProductionType'] == 'Hydro Water Reservoir') & (
                        ts_agpt['MapCode'] == reg), 'ActualGenerationOutput']
                df_tmppspgen = ts_agpt.loc[(ts_agpt['ProductionType'] == 'Hydro Pumped Storage') & (
                        ts_agpt['MapCode'] == reg), 'ActualGenerationOutput']
                df_tmppspcon = ts_agpt.loc[(ts_agpt['ProductionType'] == 'Hydro Pumped Storage') & (
                        ts_agpt['MapCode'] == reg), 'ActualConsumption']
                df_tmpfile[f'ror_{reg}'] = df_tmpror.drop_duplicates()
                df_tmpfile[f'res_{reg}'] = df_tmpres.drop_duplicates()
                df_tmpfile[f'psp_gen_{reg}'] = df_tmppspgen.drop_duplicates()
                df_tmpfile[f'psp_con_{reg}'] = df_tmppspcon.drop_duplicates()
            df_ror = df_ror.append(df_tmpfile)
            del df_tmpror, df_tmpfile

    df_ror = df_ror.sort_index()

    for reg in zones:
        df_ror.loc[:, [f'ror_{reg}']] = df_ror.loc[:, [f'ror_{reg}']].interpolate('linear')
    df_ror = df_ror.fillna(0)
    # resample to hourly frequency
    df_ror_hr = df_ror.resample('H').mean()
    df_ror_hr = df_ror_hr.interpolate('linear')
    df_ror_hr.to_csv(root_dir / 'data' / 'processed' / 'generation_hydro.csv')


def compile_reservoir_filling(root_dir, zones):

    resfill_dir = root_dir / 'data' / 'raw' / 'AggregatedFillingRateOfWaterReservoirsAndHydroStoragePlants_16.1.D'

    df_resfill = pd.DataFrame()
    for file in os.listdir(resfill_dir):
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith('.csv'):
            # read data
            df_fill = pd.read_csv(resfill_dir / filename, sep='\t', encoding='utf-8')
            df_fill.index = pd.DatetimeIndex(df_fill['DateTime'])
            df_fillreg = pd.DataFrame(columns=zones)
            for reg in zones:
                df_fillreg[f'{reg}'] = df_fill.loc[df_fill['MapCode'] == reg, 'StoredEnergy'].drop_duplicates()

            df_resfill = df_resfill.append(df_fillreg)

    df_resfill = df_resfill.sort_index()

    # eliminate data errors for Austrian reservoirs filled below 200000
    df_resfill.loc[df_resfill['AT'] < 200000, 'AT'] = None
    df_resfill = df_resfill.interpolate(method='pchip')

    df_resfill.to_csv(root_dir / 'data' / 'processed' / 'reservoir_filling.csv')


def process_energy_balance_de(root_dir):
    """
    processes annual German energy balances from AGEB into a single multi-dimensional file
    :param root_dir: Path-object giving path to project root directory. Expects subfolders 'data'/'raw' and
    'data'/'processed' to exist. Moreover, 'data'/'raw' is expected to hold German energy balances named *enbal_DE*
    :return:
    """
    dir_nblde = root_dir / 'data' / 'raw'
    nbal_de = pd.DataFrame()
    for file in os.listdir(dir_nblde):
        filename = os.fsdecode(file)

        if 'enbal_DE' in filename:
            nb = pd.read_excel(dir_nblde / filename, sheet_name='tj', header=[0], index_col=[0], na_values='')
            # generate multiindex columns
            year = int(nb.index[0][-4:])
            column_label1 = []
            column_label2 = []

            for n in list(nb.columns[1:]):
                if 'Unnamed' not in n:
                    fuel = n.strip()
                column_label1.append(fuel)
                col = nb[n]
                if not pd.isna(col[2]):
                    k = 2
                else:
                    k = 3
                label = col[k].strip().replace('\n', '').replace('-', '')
                i = k + 1
                while (not pd.isna(col[i])) & (not isinstance(col[i], int)):
                    label = label + col[i].strip().replace('-', '')
                    i += 1
                column_label2.append(label)

            columns = [f'{l1}-{l2}' for l1, l2 in zip(column_label1, column_label2)]
            nb = nb.drop(labels='Zeile', axis=1)
            nb = nb.iloc[6:, :]
            nb.columns = pd.MultiIndex.from_product([[year], columns])
            # concat df for each year
            nbal_de = pd.concat([nbal_de, nb], axis=1, join='outer')

    nbal_de = nbal_de.stack(1).swaplevel()
    nbal_de.to_csv(root_dir / 'data' / 'processed' / 'Energiebilanz_DE_TJ.csv', sep=';', decimal=',')
    logging.info(f'German energy balances processed and written to '
                 f'{root_dir / "data" / "processed" / "Energiebilanz_DE_TJ.csv"}')


def process_profiles(root_dir, zones, eta=0.9):

    capacities  = root_dir / 'data' / 'raw' / 'capacities.csv'
    opsd_timeseries = root_dir / 'data' / 'raw' / 'time_series_60min_singleindex.csv'
    hydro_generation = root_dir / 'data' / 'processed' / 'generation_hydro.csv'
    nrg_balance_at = root_dir / 'data' / 'raw' / 'enbal_AT.xlsx'
    nrg_balance_de = root_dir / 'data' / 'processed' / 'Energiebilanz_DE_TJ.csv'
    jahresreihen_eca = root_dir / 'data' / 'raw' / 'BStGes-JR1_Bilanz.xlsx'
    zeitreihen_ee_de = root_dir / 'data' / 'raw' / 'zeitreihen-ee-in-de-1990-2021-excel-en.xlsx'
    reservoir_fill = root_dir / 'data' / 'processed' / 'reservoir_filling.csv'
    profile_file = root_dir / 'data' / 'processed' / 'profiles_inflows_load.csv'

    idx = pd.IndexSlice
    caps = pd.read_csv(capacities, index_col=[0, 1, 2, 3])
    itm_caps = caps.loc[idx['Installed Capacity Out', zones, :, 'el'], ['pv', 'ror', 'wind_off', 'wind_on']]
    itm_caps.index = itm_caps.index.droplevel(0)
    itm_caps = itm_caps.unstack([0, 2])
    itm_caps.index = pd.to_datetime(itm_caps.index.values + 1, format='%Y', utc='true') - pd.Timedelta(days=184)

    ts_opsd = pd.read_csv(opsd_timeseries, index_col=0)
    ts_opsd.index = pd.DatetimeIndex(ts_opsd.index).tz_convert('utc')
    ts_opsd = ts_opsd.rename(columns={'DE_LU_price_day_ahead': 'DE_price_day_ahead'})
    ts = pd.DataFrame(index=ts_opsd.index)

    # historical day ahead prices
    for reg in zones:
        if ts_opsd.columns.str.contains(f'{reg}_price_day_ahead').any():
            ts[f'{reg}-price-day_ahead'] = ts_opsd[f'{reg}_price_day_ahead']
        else:
            ts[f'{reg}-price-day_ahead'] = np.nan

    # historical pv capacity and generation in GW(h)
    for reg in zones:
        ts[f'{reg}-pv-gen'] = ts_opsd[f'{reg}_solar_generation_actual'] / 1000
        if ts_opsd.columns.str.contains(f'{reg}_solar_capacity').any():
            ts[f'{reg}-pv-cap'] = ts_opsd[f'{reg}_solar_capacity'].fillna(method='ffill') / 1000
        else:
            ts[f'{reg}-pv-cap'] = itm_caps.loc[:, idx['pv', reg]]
            ts[f'{reg}-pv-cap'] = ts[f'{reg}-pv-cap'].interpolate()

    # historical wind onshore capacity and generation
    for reg in zones:
        ts[f'{reg}-wind_on-gen'] = ts_opsd[f'{reg}_wind_onshore_generation_actual'] / 1000
        if ts_opsd.columns.str.contains(f'{reg}_solar_capacity').any():
            ts[f'{reg}-wind_on-cap'] = ts_opsd[f'{reg}_wind_onshore_capacity'].fillna(method='ffill') / 1000
        else:
            ts[f'{reg}-wind_on-cap'] = itm_caps.loc[:, idx['wind_on', reg]]
            ts[f'{reg}-wind_on-cap'] = ts[f'{reg}-wind_on-cap'].interpolate()

    # historical wind offshore capacity and generation
    for reg in zones:
        if ts_opsd.columns.str.contains(f'{reg}_wind_offshore_generation_actual').any():
            ts[f'{reg}-wind_off-gen'] = ts_opsd[f'{reg}_wind_offshore_generation_actual'] / 1000
        else:
            ts[f'{reg}-wind_off-gen'] = 0
        if ts_opsd.columns.str.contains(f'{reg}_wind_offshore_capacity').any():
            ts[f'{reg}-wind_off-cap'] = ts_opsd[f'{reg}_wind_offshore_capacity'].fillna(method='ffill') / 1000
        else:
            ts[f'{reg}-wind_off-cap'] = 0

    # historical run-of-river capacity and generation
    ts_hydro_generation = pd.read_csv(hydro_generation, index_col=[0])
    ts_hydro_generation.index = pd.DatetimeIndex(ts_hydro_generation.index).tz_localize('utc')

    for reg in zones:
        ts[f'{reg}-ror-gen'] = ts_hydro_generation[f'ror_{reg}'] / 1000
        ts[f'{reg}-ror-cap'] = itm_caps.loc[:, idx['ror', reg]]
        ts[f'{reg}-ror-cap'] = ts[f'{reg}-ror-cap'].interpolate()

    ts[('DE-hydro-gen')] = ts_hydro_generation[['ror_DE', 'res_DE']].sum(axis=1) / 1000 + \
                              0.186 * ts_hydro_generation['psp_gen_DE'] / 1000
    # German pumped storages with natural inflows (18.6 % of installed PSP capacity, according to
    # https://www.fwt.fichtner.de/userfiles/fileadmin-fwt/Publikationen/WaWi_2017_10_Heimerl_Kohler_PSKW.pdf) are included
    # in official hydropower generation numbers.

    # historical electricity load
    for reg in zones:
        ts[f'{reg}-power-load'] = ts_opsd[f'{reg}_load_actual_entsoe_transparency'] / 1000

    # calculate scaling factor to match energy balances
    # ---
    # scaling factor for Austria
    cols = ['pv', 'wind_on', 'wind_off', 'ror', 'store', 'load']
    first_year = max(ts.index.year.min(), 2015)
    last_year = min(ts.index.year.max(), 2022)
    scale_index = pd.MultiIndex.from_product([cols, [str(yr) for yr in range(first_year, last_year)]])
    scaling_factor = pd.DataFrame(data=1, columns=zones, index=scale_index)

    nbal_at = {}
    nbal_at['cons'] = pd.read_excel(nrg_balance_at, sheet_name='Elektrische Energie',
                                    header=196, index_col=[0], nrows=190, na_values=['-']).astype('float').dropna(
        axis=0, how='all')
    nbal_at['pv'] = pd.read_excel(nrg_balance_at, sheet_name='Photovoltaik', header=196, index_col=[0], nrows=1,
                                  na_values=['-']).astype('float').dropna(axis=1)
    nbal_at['wind_on'] = pd.read_excel(nrg_balance_at, sheet_name='Wind', header=196, index_col=[0], nrows=1,
                                       na_values=['-']).astype('float').dropna(
        axis=1)
    nbal_at['hydro_eca'] = pd.read_excel(jahresreihen_eca, sheet_name='Erz', header=[8, 9], index_col=[0], nrows=37)
    nbal_at['pump_eca'] = pd.read_excel(jahresreihen_eca, sheet_name='Bil',  header=[7, 8], index_col=[0], nrows=37,
                                        na_values=['', ' '])
    nbal_at['hydro_eca'].replace(to_replace='-', value=np.nan, inplace=True)
    nbal_at['hydro'] = nbal_at['cons'].loc['aus Wasserkraft', :].sum()

    for year in range(first_year, last_year):
        if ts.loc[str(year), 'AT-pv-gen'].sum() > 0:
            scaling_factor.loc[idx['pv', str(year)], 'AT'] = (nbal_at['pv'].loc[:, year].values / 1000) / \
                                                             ts.loc[str(year), 'AT-pv-gen'].sum()
        if ts.loc[str(year), 'AT-wind_on-gen'].sum() > 0:
            scaling_factor.loc[idx['wind_on', str(year)], 'AT'] = (nbal_at['wind_on'].loc[:, year].values / 1000) / \
                                                                  ts.loc[str(year), 'AT-wind_on-gen'].sum()
        if ts.loc[str(year), 'AT-ror-gen'].sum() > 0:
            scaling_factor.loc[idx['ror', str(year)], 'AT'] = \
                nbal_at['hydro_eca'].loc[year, ('Laufkraft\nwerke', 'GWh')] * (nbal_at['hydro'][year] / 1000) / \
                nbal_at['hydro_eca'].loc[year, ('Summe\nWasser\nkraft', 'GWh')] / \
                ts.loc[str(year), 'AT-ror-gen'].sum()

        if ts_hydro_generation.loc[str(year), ['res_AT', 'psp_gen_AT']].sum().sum() > 0:
            scaling_factor.loc[idx['store', str(year)], 'AT'] = \
                nbal_at['hydro_eca'].loc[year, ('Speicher\nkraftwerke', 'GWh')] * (nbal_at['hydro'][year] / 1000) / \
                nbal_at['hydro_eca'].loc[year, ('Summe\nWasser\nkraft', 'GWh')] / \
                (ts_hydro_generation.loc[str(year), ['res_AT', 'psp_gen_AT']].sum().sum() / 1000)

        if ts.loc[str(year), 'AT-power-load'].sum() > 0:
            scaling_factor.loc[idx['load', str(year)], 'AT'] = \
                (nbal_at['cons'].loc[['Energetischer Endverbrauch', 'Transportverluste',
                                      'Verbrauch des Sektors Energie'], year].sum() / 1000 -
                 nbal_at['pump_eca'].loc[year, ('Verbrauch\nfür Pump\nspeicher', 'GWh')]) / \
                ts.loc[str(year), 'AT-power-load'].sum()

    # scaling factor for Germany
    res_de = pd.read_excel(zeitreihen_ee_de, sheet_name='3', header=[7], index_col=[0], nrows=13)

    # get rid of "Unnamed..." columns and set year as column name
    res_de_clean = res_de.filter(regex=r'^\d+', axis=1)
    res_de_clean_cols = res_de_clean.columns
    res_de_clean.columns = [str(x.year) for x in res_de_clean_cols]
    res_de = res_de_clean

    nbal_de = pd.read_csv(nrg_balance_de, index_col=[0, 1], sep=';', decimal=',')
    # convert from TJ to GWh
    nbal_de = nbal_de / 3.6

    for year in range(first_year, last_year):
        if ts.loc[str(year), 'DE-pv-gen'].sum() > 0:
            scaling_factor.loc[idx['pv', str(year)], 'DE'] = \
                res_de.loc['Solar Photovoltaic', str(year)] / ts.loc[str(year), 'DE-pv-gen'].sum()

        if ts.loc[str(year), 'DE-wind_on-gen'].sum() > 0:
            scaling_factor.loc[idx['wind_on', str(year)], 'DE'] = \
                res_de.loc['Wind energy onshore', str(year)] / ts.loc[str(year), 'DE-wind_on-gen'].sum()

        if ts.loc[str(year), 'DE-wind_off-gen'].sum() > 0:
            scaling_factor.loc[idx['wind_off', str(year)], 'DE'] = \
                res_de.loc['Wind energy offshore', str(year)] / ts.loc[str(year), 'DE-wind_off-gen'].sum()

        if ts.loc[str(year), 'DE-hydro-gen'].sum() > 0:
            scaling_factor.loc[idx['ror', str(year)], 'DE'] = \
                res_de.loc['Hydropower 1)', str(year)] / \
                ts.loc[str(year), 'DE-hydro-gen'].sum()

        if ts.loc[str(year), 'DE-power-load'].sum() > 0:
            scaling_factor.loc[idx['load', str(year)], 'DE'] = \
                nbal_de.loc[idx['Elektrischer Strom und-Strom',
                                   ['ENDENERGIEVERBRAUCH', 'Fackel- u. Leitungsverluste']], str(year)].sum() / \
                ts.loc[str(year), 'DE-power-load'].sum()

    # generate scaled generation profiles and electric load
    intermittents = {'pv': 'Solar', 'wind_on': 'Wind_On', 'wind_off': 'Wind_off', 'ror': 'Water'}
    for reg in zones:
        for yr in range(first_year, last_year):
            ts.loc[str(yr), f'{reg}-power-load'] = ts.loc[str(yr), f'{reg}-power-load'] * \
                                                   scaling_factor.loc[idx['load', str(yr)], reg]

            for itm, fuel in intermittents.items():
                if scaling_factor.loc[idx[itm, str(yr)], reg] < 2:
                    ts.loc[str(yr), f'{reg}-{fuel}-profile'] = ts.loc[str(yr), f'{reg}-{itm}-gen'] / \
                                                               ts.loc[str(yr), f'{reg}-{itm}-cap'] * \
                                                               scaling_factor.loc[idx[itm, str(yr)], reg]
                else:
                    ts.loc[str(yr), f'{reg}-{fuel}-profile'] = ts.loc[str(yr), f'{reg}-{itm}-gen'] / \
                                                               ts.loc[str(yr), f'{reg}-{itm}-cap']

    # TODO: wind offshore capacity from opsd is wrong -- needs to be corrected

    # ----- approximate reservoir inflows -----
    # hourly reservoir filling levels
    df_hydro_fill = pd.read_csv(reservoir_fill, index_col=[0])
    df_hydro_fill.index = pd.DatetimeIndex(df_hydro_fill.index).tz_localize('utc')
    """
    # resample storage fill to hourly frequency
    ts_hydro_fill = pd.DataFrame(
        index=pd.date_range(f'{df_hydro_fill.head(1).index.year[0]}/01/01-00:00',
                            f'{df_hydro_fill.tail(1).index.year[0]}/12/31-23:00', freq='h', tz='utc'), columns=zones)
    ts_hydro_fill.update(df_hydro_fill[zones].resample('H').interpolate(method='pchip'))
    ts_hydro_fill = ts_hydro_fill.fillna(method='pad')
    ts_hydro_fill = ts_hydro_fill.fillna(method='bfill')
    for reg in zones:
        ts[(f'{reg}', 'fill-reservoir-entsoe')] = ts_hydro_fill.loc[:, reg] / df_hydro_fill[reg].max()
    """
    # reservoir inflows
    inflows = pd.DataFrame(columns=zones)
    for reg in zones:
        # upsample turbining and pumping to fill rate times and calculate balance at time of fill readings
        inflows[reg] = (df_hydro_fill[reg] - df_hydro_fill[reg].shift(periods=-1) -
                        ts_hydro_generation[f'psp_con_{reg}'].resample('W-MON').sum() * eta +
                        ts_hydro_generation[f'psp_gen_{reg}'].resample('W-MON').sum() / eta +
                        ts_hydro_generation[f'res_{reg}'].resample('W-MON').sum() / eta ) / 1000 / 168
        # shift inflow estimate up to avoid negative inflows
        if inflows[reg].min() < 0:
            inflows[reg] = inflows[reg] - inflows[reg].min() * 1.1
        # upsample inflows
        inflows = inflows.resample('H').interpolate(method='pchip')

    # inflow correction factor
    inflows_nbal = (nbal_at['hydro_eca'][('Speicher\nkraftwerke', 'GWh')] / eta -
                    nbal_at['pump_eca'][('Verbrauch\nfür Pump\nspeicher', 'GWh')] * eta) * \
                   (nbal_at['hydro_eca'][('Summe\nWasser\nkraft', 'GWh')] / (nbal_at['hydro'] / 1000))
    inflows_annual = inflows.resample('Y').sum()
    inflows_annual.index = inflows_annual.index.year
    # scale inflows
    for yr in range(first_year, last_year):
        inflows.loc[str(yr)] = inflows.loc[str(yr)] * inflows_nbal.loc[yr] / inflows_annual.loc[yr, 'AT']

    for reg in zones:
        ts.loc[:, f'{reg}-reservoir-inflows'] = inflows[reg]
        ts.loc[ts.loc[:, f'{reg}-reservoir-inflows'] < 0, f'{reg}-reservoir-inflows'] = 0

    # save data
    ts.to_csv(profile_file, sep=';', decimal=',')
    logging.info(f'Renewables profiles, inflows and load processed and saved to {profile_file}')
    return ts


def do_processing(root_dir, years, zones):
    """
    processes data stored in root_dir/data/raw and saves it to root_dir/data/processed. Intended for use with power
    system model medea.
    :param root_dir:
    :param years:
    :param zones:
    :return:
    """
    setup_logging()

    # file paths
    root_dir = Path(root_dir)
    ERA_DIR = root_dir / 'data' / 'raw' / 'era5'
    # imf_file = root_dir / 'data' / 'raw' / 'imf_price_data.xlsx'
    ngas_file = root_dir / 'data' / 'raw' / 'egas_aufkommen_export_1991.xlsm'
    brent_file = root_dir / 'data' / 'raw' / 'RBRTEm.xls'
    coal_file = root_dir / 'data' / 'raw' / 'energiepreisentwicklung_5619001.xlsx'
    fx_file = root_dir / 'data' / 'raw' / 'ecb_fx_data.csv'
    co2_file = root_dir / 'data' / 'raw' / 'eua_price.csv'
    enbal_at = root_dir / 'data' / 'raw' / 'enbal_AT.xlsx'
    enbal_de = root_dir / 'data' / 'processed' / 'Energiebilanz_DE_TJ.csv'
    PPLANT_DB = root_dir / 'data' / 'raw' / 'conventional_power_plants_EU.csv'

    process_dir = root_dir / 'data' / 'processed'
    fuel_price_file = process_dir / 'monthly_fuel_prices.csv'
    co2_price_file = process_dir / 'co2_price.csv'
    MEAN_TEMP_FILE = process_dir / 'temp_daily_mean.csv'
    heat_cons_file = process_dir / 'heat_hourly_consumption.csv'
    ts_file = process_dir / 'time_series.csv'

    package_dir = Path(sysconfig.get_path('data'))
    CONSUMPTION_PATTERN = package_dir / 'raw' / 'consumption_pattern.csv'

    if not os.path.exists(process_dir):
        os.makedirs(process_dir)

    # process PRICE data
    # df_imf = pd.read_excel(imf_file, index_col=[0], skiprows=[1, 2, 3])
    # df_imf.index = pd.to_datetime(df_imf.index, format='%YM%m')
    df_ngas = pd.read_excel(ngas_file)
    p_ngas = pd.DataFrame(data=0, index=pd.date_range(start='2010/01/01', end='2021/12/31', freq='MS'), columns=['Preis'])
    p_ngas.loc['2010', 'Preis'] = np.round(df_ngas.iloc[112:124, 2].astype('float') / 277.7777, 2).values
    p_ngas.loc['2011', 'Preis'] = np.round(df_ngas.iloc[81:93, 10].astype('float') / 277.7777, 2).values
    p_ngas.loc['2012', 'Preis'] = np.round(df_ngas.iloc[81:93, 8].astype('float') / 277.7777, 2).values
    p_ngas.loc['2013', 'Preis'] = np.round(df_ngas.iloc[81:93, 6].astype('float') / 277.7777, 2).values
    p_ngas.loc['2014', 'Preis'] = np.round(df_ngas.iloc[81:93, 4].astype('float') / 277.7777, 2).values
    p_ngas.loc['2015', 'Preis'] = np.round(df_ngas.iloc[81:93, 2].astype('float') / 277.7777, 2).values
    p_ngas.loc['2016', 'Preis'] = np.round(df_ngas.iloc[44:56, 10].astype('float') / 277.7777, 2).values
    p_ngas.loc['2017', 'Preis'] = np.round(df_ngas.iloc[44:56, 8].astype('float') / 277.7777, 2).values
    p_ngas.loc['2018', 'Preis'] = np.round(df_ngas.iloc[44:56, 6].astype('float') / 277.7777, 2).values
    p_ngas.loc['2019', 'Preis'] = np.round(df_ngas.iloc[44:56, 4].astype('float') / 277.7777, 2).values
    p_ngas.loc['2020', 'Preis'] = np.round(df_ngas.iloc[10:22, 7].astype('float') / 277.7777, 2).values
    p_ngas.loc['2021', 'Preis'] = np.round(df_ngas.iloc[10:22, 3].astype('float') / 277.7777, 2).values

    p_brent = pd.read_excel(brent_file, sheet_name='Data 1', index_col=[0], skiprows=[0, 1])
    p_brent.index = p_brent.index.to_period('M').to_timestamp()

    df_coal = pd.read_excel(coal_file, sheet_name='5.1 Steinkohle und Braunkohle', index_col=[0],
                            skiprows=[0, 1, 2, 3, 4, 5])
    df_coal = df_coal.iloc[5:17, 0:12].stack().astype('float')
    p_coal = pd.DataFrame(data=0, index=pd.date_range(start='2010/01/01', end='2021/12/31', freq='MS'), columns=['Preis'])
    p_coal.loc['2010':'2021', 'Preis'] = df_coal.values * 67.9 / 100  # Euro pro Tonne SKE

    df_fx = pd.read_csv(fx_file, index_col=[0], skiprows=[0, 2, 3, 4, 5], usecols=[1], na_values=['-']).astype('float')
    df_fx.index = pd.to_datetime(df_fx.index, format='%Y-%m-%d')

    # convert prices to EUR per MWh
    df_prices_mwh = pd.DataFrame(index=p_coal.index, columns=['USD_EUR', 'Oil', 'Coal', 'Gas'])
    df_prices_mwh['USD_EUR'] = df_fx.resample('MS').mean()
    df_prices_mwh['Oil'] = (p_brent.loc['2010':'2021', 'Europe Brent Spot Price FOB (Dollars per Barrel)'] /
                                 df_prices_mwh['USD_EUR']) * 7.52 / 11.63
    df_prices_mwh['Coal'] = p_coal / 8.141
    df_prices_mwh['Gas'] = p_ngas  # df_imf['PNGASEU'] / df_prices_mwh['USD_EUR'] / 0.29307
    # drop rows with all nan
    df_prices_mwh.dropna(how='all', inplace=True)

    df_prices_mwh.to_csv(fuel_price_file)
    logging.info(f'fuel prices processed and saved to {fuel_price_file}')

    df_price_co2 = pd.read_csv(co2_file, index_col=[0])
    df_price_co2.index = pd.to_datetime(df_price_co2.index, format='%Y-%m-%d')
    df_price_co2 = df_price_co2.rename(columns={'Settle': 'EUA'})
    df_price_co2['EUA'].to_csv(co2_price_file)
    logging.info(f'CO2 prices processed and saved to {co2_price_file}')

    # process temperature data
    # get coordinates of co-gen plants
    db_plants = pd.read_csv(PPLANT_DB)
    daily_mean_temp = mean_temp_at_plants(db_plants, ERA_DIR, years, zones)
    daily_mean_temp.to_csv(MEAN_TEMP_FILE)
    logging.info(f'Temperatures processed and saved to {MEAN_TEMP_FILE}')

    # process German energy balances
    process_energy_balance_de(root_dir)

    # process HEAT LOAD
    # read German energy balances
    nbal_de = pd.read_csv(enbal_de, sep=';', index_col=[0, 1])
    ht_enduse_de = nbal_de.loc[pd.IndexSlice['Elektrischer Strom und-Fernwärme', :], :] / 3.6
    ht_enduse_de.index = ht_enduse_de.index.get_level_values(1)

    # process Austrian energy balances
    ht_gen_at = pd.read_excel(enbal_at, sheet_name='Fernwärme', index_col=[0], header=[196], nrows=190)
    ht_gen_at = ht_gen_at.loc[['Energetischer Endverbrauch', 'Transportverluste'], years].sum() / 1000
    ht_enduse_at = pd.read_excel(enbal_at, sheet_name='Fernwärme', header=[438], index_col=[0], nrows=24,
                                 na_values=['-']).astype('float')
    ht_enduse_at = ht_enduse_at / 1000
    ht_mult_at = ht_gen_at / ht_enduse_at.loc[['Private Haushalte', 'Öffentliche und Private Dienstleistungen',
                                               'Produzierender Bereich'], years].sum()

    ht_cons = pd.DataFrame(index=years,
                           columns=pd.MultiIndex.from_product([zones, ['HE08', 'HM08', 'HG08', 'WW', 'IND']]))
    ht_cons.loc[years, ('AT', 'HE08')] = ht_enduse_at.loc['Private Haushalte', years] * ht_mult_at * 0.376 * 0.75
    ht_cons.loc[years, ('AT', 'HM08')] = ht_enduse_at.loc['Private Haushalte', years] * ht_mult_at * 0.624 * 0.75
    ht_cons.loc[years, ('AT', 'WW')] = ht_enduse_at.loc['Private Haushalte', years] * ht_mult_at * 0.25
    ht_cons.loc[years, ('AT', 'HG08')] = ht_enduse_at.loc['Öffentliche und Private Dienstleistungen', years] * ht_mult_at
    ht_cons.loc[years, ('AT', 'IND')] = ht_enduse_at.loc['Produzierender Bereich', years] * ht_mult_at
    ht_cons.loc[years, ('DE', 'HE08')] = ht_enduse_de.loc['Haushalte', years] * 0.376 * 0.75
    ht_cons.loc[years, ('DE', 'HM08')] = ht_enduse_de.loc['Haushalte', years] * 0.624 * 0.75
    ht_cons.loc[years, ('DE', 'WW')] = ht_enduse_de.loc['Haushalte', years] * 0.25
    ht_cons.loc[years, ('DE', 'HG08')] = ht_enduse_de.loc[
        'Gewerbe, Handel, Dienstleistungen u.übrige Verbraucher', years]
    ht_cons.loc[years, ('DE', 'IND')] = ht_enduse_de.loc[
        'Bergbau, Gew. Steine u. Erden, Verarbeit. Gewerbe insg.', years]

    """ 
    Above transformations are based on following Assumptions
     * share of heating energy used for hot water preparation: 25%
       (cf. https://www.umweltbundesamt.at/fileadmin/site/publikationen/REP0074.pdf, p. 98)
     * share of heat delivered to single-family homes:
       - in Austria 2/3 of houses are single-family houses, which are home to 40% of the population
       - in Germany, 65.1% of houses are EFH, 17.2% are ZFH and 17.7% are MFH
         (cf. https://www.statistik.rlp.de/fileadmin/dokumente/gemeinschaftsveroeff/zen/Zensus_GWZ_2014.pdf)
       - average space in single-family houses: 127.3 m^2; in multiple dwellings: 70.6 m^2
       - specific heat consumption: EFH: 147.9 kWh/m^2; MFH: 126.5 kWh/m^2
         (cf. http://www.rwi-essen.de/media/content/pages/publikationen/rwi-projektberichte/
         PB_Datenauswertung-Energieverbrauch-privHH.pdf, p. 5)
       - implied share of heat consumption, assuming on average 7 appartments per MFH: 37.6% EFH; 62.4% MFH
    """
    # Accounting for own consumption and pipe losses
    ht_cons = ht_cons.multiply(1.125)

    df_heat = pd.read_csv(MEAN_TEMP_FILE, index_col=[0], parse_dates=True)
    df_heat['year'] = df_heat.index.year
    df_heat['weekday'] = df_heat.index.strftime('%a')
    df_heat.fillna(method='pad', inplace=True)  # fill NA to prevent NAs in temperature smoothing below

    cons_pattern = pd.read_csv(CONSUMPTION_PATTERN, index_col=[0, 1])
    cons_pattern = cons_pattern.rename_axis('hour', axis=1)
    cons_pattern = cons_pattern.unstack('consumer').stack('hour')

    ht_consumption = heat_consumption(zones, years, ht_cons, df_heat, cons_pattern)
    ht_consumption.to_csv(heat_cons_file)
    logging.info(f'exported hourly heat demand to {heat_cons_file}')

    # process time series data
    compile_hydro_generation(root_dir, zones)
    # legacy code: d:/git_repos/medea_data_atde_local/src/compile/compile_timeseries.py
    ts = process_profiles(root_dir, zones, eta=0.9)
    logging.info(f'time series processed')

    tsx = ht_consumption.groupby(axis=1, level=0).sum().merge(df_prices_mwh.resample('H').interpolate('pchip'),
                                                              left_index=True, right_index=True, how='outer')
    tsx = tsx.rename(columns={'AT': 'AT-heat-load', 'DE': 'DE-heat-load'})
    tsx = tsx.merge(df_price_co2['EUA'].resample('H').interpolate('pchip'),
                    left_index=True, right_index=True, how='outer')
    #tsx = tsx.merge(ht_consumption.groupby(axis=1, level=0).sum(), left_index=True, right_index=True, how='outer')
    tsx = tsx.tz_localize('UTC')
    tsx = tsx.merge(ts, left_index=True, right_index=True, how='outer')
    tsx.to_csv(ts_file)
    logging.info('data processing completed')

