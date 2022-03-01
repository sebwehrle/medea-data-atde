# %% imports
from pathlib import Path
import sysconfig
import pandas as pd
import numpy as np
from medea_data_atde.retrieve import hours_in_year


def date2set_index(df, year, tz='utc', set_sym='h'):
    df['DateTime'] = pd.to_datetime(df.index)
    df.set_index('DateTime', inplace=True)
    start_stamp = pd.Timestamp(year, 1, 1, 0, 0).tz_localize(tz)
    end_stamp = pd.Timestamp(year, 12, 31, 23, 0).tz_localize(tz)
    df = df.loc[(start_stamp <= df.index) & (df.index <= end_stamp)]
    if len(df) == hours_in_year(year):
        hour_index = [f'{set_sym}{hour}' for hour in range(1, hours_in_year(year) + 1)]
        hour_index = pd.DataFrame(data=hour_index, columns=['hours'])
        df.set_index(hour_index['hours'], inplace=True)
    else:
        raise ValueError('Mismatch of time series data and model time resolution. Is year wrong?')
    return df


def compile_symbols(root_dir, timeseries, zones, year, invest_conventionals=True, invest_renewables=True,
                    invest_storage=True, invest_tc=True):
    """
    prepares dictionaries used to build input data gdx-files for power system model medea
    :param root_dir: root directory
    :param timeseries: path to timeseries_regional.csv
    :param zones: ISO 2-letter country code for countries to model. Default: ['AT', 'DE']
    :param year: integer of year to model. Default: 2016
    :param invest_conventionals: boolean
    :param invest_renewables: boolean
    :param invest_storage: boolean
    :param invest_tc: boolean for investment in transmission capacity
    :return:
    """

    idx = pd.IndexSlice
    package_dir = Path(sysconfig.get_path('data'))
    technologies = {
        'capacity': pd.read_csv(package_dir / 'raw' / 'capacities.csv', index_col=[0, 1, 2, 3]),
        'capacity_transmission': pd.read_csv(package_dir / 'raw' / 'transmission.csv', index_col=[0, 1]),
        'technology': pd.read_csv(package_dir / 'raw' / 'technologies.csv', index_col=[0]).dropna(axis=0, how='all'),
        'operating_region': pd.read_csv(package_dir / 'raw' / 'operating_region.csv', index_col=[0, 1, 2]),
    }

    ts_data = {
        'timeseries': pd.read_csv(timeseries, index_col=[0])
    }

    estimates = {
        'external_cost': pd.read_csv(package_dir / 'raw' / 'external_cost.csv', index_col=[0]),
        'point_estimates': pd.read_csv(package_dir / 'raw' / 'point_estimates.csv', index_col=[0]),
        'price_nonmarket_fuels': pd.read_csv(package_dir / 'raw' / 'price_nonmarket_fuels.csv', index_col=[0]),
    }

    # create SETS
    # --------------------------------------------------------------------------- #
    sets = {
        'e': [carrier for carrier in np.unique(technologies['technology'][['fuel', 'primary_product']])],
        # all energy carriers
        'i': [input for input in technologies['technology']['fuel'].unique()],  # energy inputs
        'f': [final for final in technologies['technology']['primary_product'].unique()],  # final energy
        't': [tec for tec in technologies['technology'].index.unique()],  # technologies
        'c': [chp for chp in technologies['operating_region'].index.get_level_values(0).unique()],
        # co-generation technologies
        'd': [plant for plant in technologies['technology'].loc[
            technologies['technology']['conventional'] == 1].index.unique()],  # dispatchable technologies
        'r': [intmit for intmit in technologies['technology'].loc[
            technologies['technology']['intermittent'] == 1].index.unique()],  # intermittent technologies
        's': [storage for storage in technologies['technology'].loc[
            technologies['technology']['storage'] == 1].index.unique()],  # storage technologies
        'g': [transmit for transmit in technologies['technology'].loc[
            technologies['technology']['transmission'] == 1].index.unique()],  # transmission technologies
        'l': [f'l{x}' for x in range(1, 5)],  # feasible operating regions
        'h': [f'h{hour}' for hour in range(1, hours_in_year(year) + 1)],  # time steps / hours
        'z': [zone for zone in zones]  # market zones
    }

    # convert set-dictionaries to DataFrames
    for key, value in sets.items():
        sets.update({key: pd.DataFrame(data=sets[key], columns=['uni_0'])})

    # create PARAMETERS
    # --------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------- #
    # ** process technology data **
    # amend plant data by co-generation fuel need
    technologies['operating_region']['fuel_need'] = technologies['operating_region']['fuel'] / \
                                                    technologies['technology'].loc[
                                                        technologies['technology'][
                                                            'heat_generation'] == 1, 'eta_ec']
    # transmission distances
    technologies['distance'] = technologies['capacity_transmission']['distance'].loc[
                               technologies['capacity_transmission']['distance'].index.get_level_values(0).str.contains(
                                   '|'.join(zones)) &
                               technologies['capacity_transmission']['distance'].index.get_level_values(1).str.contains(
                                   '|'.join(zones)), :]
    # transmission capacities
    technologies['capacity_transmission'] = technologies['capacity_transmission'].assign(f='el').set_index('f',
                                                                                                           append=True)
    technologies.update({'capacity_transmission': technologies['capacity_transmission']['ATC'].loc[
                                                  technologies['capacity_transmission']['ATC'].index.get_level_values(
                                                      0).str.contains('|'.join(zones)) &
                                                  technologies['capacity_transmission']['ATC'].index.get_level_values(
                                                      1).str.contains('|'.join(zones)), :] / 1000})

    # process time series data
    # --------------------------------------------------------------------------- #
    ts_data['zonal'] = ts_data['timeseries'].loc[:,
                       ts_data['timeseries'].columns.str.startswith(tuple(zones))].copy()
    ts_data['zonal'].columns = ts_data['zonal'].columns.str.split('-', expand=True)
    # adjust column naming to reflect proper product names ('el' and 'ht')
    ts_data['zonal'] = ts_data['zonal'].rename(columns={'power': 'el', 'heat': 'ht'})

    # date-time conversion and selection
    ts_data['zonal'] = date2set_index(ts_data['zonal'], year)
    ts_data['timeseries'] = date2set_index(ts_data['timeseries'], year)

    # process PRICES
    # create price time series incl transport cost
    ts_data['timeseries'].loc[:, 'Nuclear'] = estimates['price_nonmarket_fuels'].loc['Nuclear', :].values[0]
    ts_data['timeseries'].loc[:, 'Lignite'] = estimates['price_nonmarket_fuels'].loc['Lignite', :].values[0]
    ts_data['timeseries'].loc[:, 'Biomass'] = estimates['price_nonmarket_fuels'].loc['Biomass', :].values[0]

    model_prices = ['Coal', 'Oil', 'Gas', 'EUA', 'Nuclear', 'Lignite', 'Biomass', 'price-day_ahead']
    ts_data['price'] = pd.DataFrame(index=ts_data['timeseries'].index,
                                    columns=pd.MultiIndex.from_product([model_prices, zones]))
    for reg in zones:
        for fuel in model_prices:
            if fuel in estimates['external_cost'].index:
                ts_data['price'][(fuel, reg)] = ts_data['timeseries'][fuel] + estimates['external_cost'].loc[
                    fuel, reg]
            elif fuel in ['price-day_ahead']:
                ts_data['price'][(fuel, reg)] = ts_data['timeseries'][f'{reg}-{fuel}']
            else:
                ts_data['price'][(fuel, reg)] = ts_data['timeseries'][fuel]

    # process INFLOWS to hydro storage plants
    hydro_storage = technologies['technology'].loc[(technologies['technology']['storage'] == 1) &
                                                   (technologies['technology']['fuel'] == 'Water')].index
    inflow_factor = technologies['capacity'].loc[idx['Installed Capacity Out', zones, year], hydro_storage].T / \
                    technologies['capacity'].loc[idx['Installed Capacity Out', zones, year], hydro_storage].T.sum()
    inflow_factor.columns = inflow_factor.columns.droplevel([0, 2, 3])
    ts_inflows = pd.DataFrame(index=list(ts_data['zonal'].index),
                              columns=pd.MultiIndex.from_product([zones,
                                                                  [s[0] for s in sets['s'].values if 'hyd' in s[0]]]))
    for zone in list(zones):
        for strg in hydro_storage:
            ts_inflows.loc[:, (zone, strg)] = ts_data['zonal'].loc[:, idx[zone, 'reservoir', 'inflows']] * \
                                              inflow_factor.loc[strg, zone]
    ts_data.update({'INFLOWS': ts_inflows})

    # --------------------------------------------------------------------------- #
    # peak load and profiles
    # --------------------------------------------------------------------------- #
    ts_data.update({'PEAK_LOAD': ts_data['zonal'].loc[:, idx[:, 'el', 'load']].max().unstack((1, 2)).squeeze()})
    peak_profile = ts_data['zonal'].loc[:, idx[:, :, 'profile']].max().unstack(2).drop('ror', axis=0, level=1)
    peak_profile.fillna(0, inplace=True)
    ts_data.update({'PEAK_PROFILE': peak_profile})

    # --------------------------------------------------------------------------- #
    # mappings
    # --------------------------------------------------------------------------- #
    map_input =  pd.DataFrame(data=False, index=technologies['technology'].index,
                              columns=np.unique(technologies['technology'][['fuel', 'primary_product']]))
    for ix, row in technologies['technology'].iterrows():
        map_input.loc[ix, row['fuel']] = True

    map_output = pd.DataFrame(data=False, index=technologies['technology'].index,
                              columns=technologies['technology']['primary_product'].unique())
    for ix, row in technologies['technology'].iterrows():
        map_output.loc[ix, row['primary_product']] = True
    map_output.loc[map_output.index.str.contains('chp'), 'ht'] = True

    # --------------------------------------------------------------------------- #
    # limits on investment - long-run vs short-run
    # --------------------------------------------------------------------------- #
    # SWITCH_INVEST
    invest_limits = {
        # 'potentials': pd.read_csv(potentials, index_col=[0]),
        'thermal': pd.DataFrame([float('inf') if invest_conventionals else 0]),
        'intermittent': pd.DataFrame(data=[float('inf') if invest_renewables else 0][0], index=zones, columns=sets['r'].index),
        'storage': pd.DataFrame(data=[float('inf') if invest_storage else 0][0], index=zones, columns=sets['s'].index),
        'atc': pd.DataFrame(data=[1 if invest_tc else 0][0], index=zones, columns=zones)
    }

    parameters = {
        'AIR_POL_COST_FIX': [['i'], estimates['external_cost']['fixed cost'].dropna(), '[EUR per MW]'],
        'AIR_POL_COST_VAR': [['i'], estimates['external_cost']['variable cost'].dropna(), '[EUR per MWh]'],
        'CAPACITY': [['z', 't'], technologies['capacity'].loc[idx['Installed Capacity Out', zones, year], :].T.stack(
            1).reorder_levels((1, 0)), '[GW]'],
        'CAPACITY_X': [['z', 'z', 'f'], technologies['capacity_transmission'], '[GW]'],
        'CAPACITY_STORAGE': [['z', 'f', 's'], technologies['capacity'].loc[
            idx['Storage Capacity', zones, year], [s[0] for s in sets['s'].values]].T.stack((1, 3)).reorder_levels((1, 2, 0)), '[GWh]'],
        'CAPACITY_STORE_IN': [['z', 'f', 's'], technologies['capacity'].loc[
            idx['Installed Capacity In', zones, year], [s[0] for s in sets['s'].values]].T.stack((1, 3)).reorder_levels((1, 2, 0)),
                              '[GW]'],
        'CAPACITY_STORE_OUT': [['z', 'f', 's'], technologies['capacity'].loc[
            idx['Installed Capacity Out', zones, year], [s[0] for s in sets['s'].values]].T.stack((1, 3)).reorder_levels((1, 2, 0)),
                               '[GW]'],
        'OVERNIGHTCOST': [['t'], technologies['technology'].loc[:, 'capex_p'].round(4), '[EUR per MW]'],
        'OVERNIGHTCOST_E': [['s'], technologies['technology'].loc[[s[0] for s in sets['s'].values], 'capex_e'], '[EUR per MWh]'],
        'OVERNIGHTCOST_P': [['t'], technologies['technology'].loc[[s[0] for s in sets['s'].values], 'capex_p'], '[EUR per MW]'],
        'OVERNIGHTCOST_X': [['g'], technologies['technology'].loc[[g[0] for g in sets['g'].values], 'capex_p'], '[EUR per MW]'],
        'CO2_INTENSITY': [['i'], estimates['external_cost']['CO2_intensity'].dropna(), '[t CO2 per MWh fuel input]'],
        'CONVERSION': [['e', 'f', 't'], technologies['technology'][['fuel', 'primary_product', 'eta_ec']].reset_index().set_index(['fuel', 'primary_product', 'set_id']),'[]'],
        'COST_OM_QFIX': [['t'], technologies['technology']['opex_f'], '[EUR per MW]'],
        'COST_OM_VAR': [['t'], technologies['technology']['opex_v'], '[EUR per MWh]'],
        'DEMAND': [['z', 'h', 'f'], ts_data['zonal'].loc[
                                    :, idx[:, :, 'load']].stack((0, 1)).reorder_levels((1, 0, 2)).round(4), '[GW]'],
        'DISCOUNT_RATE': [['z'], estimates['point_estimates'].loc['wacc', :], '[]'],
        'DISTANCE': [['z', 'z'], technologies['distance'], '[km]'],
        'FEASIBLE_INPUT': [['l', 'i', 'c'], technologies['operating_region']['fuel'].div(
            technologies['technology']['eta_ec']).reorder_levels((1, 2, 0)), '[GW]'],
        'FEASIBLE_OUTPUT': [['l', 'f', 'c'], technologies['operating_region'][
            ['el', 'ht']].droplevel('f').stack().reorder_levels((1, 2, 0)), '[GW]'],
        'INFLOWS': [['z', 'h', 's'], ts_data['INFLOWS'].stack((0, 1)).reorder_levels(
            (1, 0, 2)).astype('float').round(4), '[GW]'],
        'LAMBDA': [['z'], estimates['point_estimates'].loc['LAMBDA', :], '[]'],
        'LIFETIME': [['t'], technologies['technology']['lifetime'], '[a]'],
        'MAP_INPUTS': [['e', 't'], map_input.stack().reorder_levels((1, 0)), '[]'],
        'MAP_OUTPUTS': [['f', 't'], map_output.stack().reorder_levels((1, 0)), '[]'],
        'PEAK_LOAD': [['z'], ts_data['PEAK_LOAD'], '[GW]'],
        'PEAK_PROFILE': [['z', 'i'], ts_data['PEAK_PROFILE'], '[]'],
        'PRICE_CO2': [['z', 'h'], ts_data['price'].loc[:, idx['EUA', :]].stack().reorder_levels((1, 0)), '[EUR per t]'],
        'PRICE': [['z', 'h', 'i'], ts_data['price'].drop(columns=['EUA'], level=0).stack(
            (0, 1)).reorder_levels((2, 0, 1)).round(4), '[EUR per MWh]'],
        'PRICE_TRADE': [['i'], estimates['price_nonmarket_fuels'].loc['Syngas', :], '[EUR per MWh]'],
        'PROFILE': [['z', 'h', 'i'], ts_data['zonal'].loc[:, idx[:, :, 'profile']].stack(
            (0, 1)).reorder_levels((1, 0, 2)).round(4), '[]'],
        'SIGMA': [['z'], estimates['point_estimates'].loc['SIGMA', :], '[]'],
        'VALUE_NSE': [['z'], estimates['point_estimates'].loc['VALUE_NSE', :], '[EUR per MWh]'],
        # 'SWITCH_INVEST': invest_limits['thermal'],
    }

    for key, val in parameters.items():
        if val[0]:
            parameters[key][1] = val[1].reset_index()
        cols = [f'{i[1]}_{i[0]}' for i in enumerate(val[0])]
        cols.extend(['value'])
        parameters[key][1].columns = cols

    return sets, parameters
