# %% imports
import logging
from logging_config import setup_logging
import pandas as pd
from config import MEDEA_ROOT_DIR, ERA_DIR, COUNTRY, YEARS, zones, imf_file, fx_file, co2_file, url_ageb_bal
from src.fun_process import mean_temp_at_plants, heat_consumption

setup_logging()

# %% file paths
fuel_price_file = MEDEA_ROOT_DIR / 'data' / 'processed' / 'monthly_fuel_prices.csv'
co2_price_file = MEDEA_ROOT_DIR / 'data' / 'processed' / 'co2_price.csv'
enbal_at = MEDEA_ROOT_DIR / 'data' / 'raw' / 'enbal_AT.xlsx'
PPLANT_DB = MEDEA_ROOT_DIR / 'data' / 'processed' / 'power_plant_db.xlsx'
MEAN_TEMP_FILE = MEDEA_ROOT_DIR / 'data' / 'processed' / 'temp_daily_mean.csv'
CONSUMPTION_PATTERN = MEDEA_ROOT_DIR / 'data' / 'raw' / 'consumption_pattern.xlsx'
heat_cons_file = MEDEA_ROOT_DIR / 'data' / 'processed' / 'heat_hourly_consumption.csv'

# %% process PRICE data
df_imf = pd.read_excel(imf_file, index_col=[0], skiprows=[1, 2, 3])
df_imf.index = pd.to_datetime(df_imf.index, format='%YM%m')

df_fx = pd.read_csv(fx_file, index_col=[0], skiprows=[0, 2, 3, 4, 5], usecols=[1], na_values=['-']).astype('float')
df_fx.index = pd.to_datetime(df_fx.index, format='%Y-%m-%d')

# convert prices to EUR per MWh
df_prices_mwh = pd.DataFrame(index=df_imf.index, columns=['USD_EUR', 'Brent_UK', 'Coal_SA', 'NGas_DE'])
df_prices_mwh['USD_EUR'] = df_fx.resample('MS').mean()
df_prices_mwh['Brent_UK'] = df_imf['POILBRE'] / df_prices_mwh['USD_EUR'] * 7.52 / 11.63
df_prices_mwh['Coal_SA'] = df_imf['PCOALSA_USD'] / df_prices_mwh['USD_EUR'] / 6.97333
df_prices_mwh['NGas_DE'] = df_imf['PNGASEU'] / df_prices_mwh['USD_EUR'] / 0.29307
# drop rows with all nan
df_prices_mwh.dropna(how='all', inplace=True)

df_prices_mwh.to_csv(fuel_price_file)
logging.info(f'fuel prices processed and saved to {fuel_price_file}')

df_price_co2 = pd.read_csv(co2_file, index_col=[0])
df_price_co2.index = pd.to_datetime(df_price_co2.index, format='%Y-%m-%d')
df_price_co2['Settle'].to_csv(co2_price_file)
logging.info(f'CO2 prices processed and saved to {co2_price_file}')

# %% process temperature data
# get coordinates of co-gen plants
db_plants = pd.read_excel(PPLANT_DB)
daily_mean_temp = mean_temp_at_plants(db_plants, ERA_DIR, COUNTRY, YEARS, zones)
daily_mean_temp.to_csv(MEAN_TEMP_FILE)
logging.info(f'Temperatures processed and saved to {MEAN_TEMP_FILE}')

# %% process HEAT LOAD
# process German energy balances
ht_enduse_de = pd.DataFrame()
for yr in [x - 2000 for x in YEARS]:
    enebal_de = MEDEA_ROOT_DIR / 'data' / 'raw' / f'enbal_DE_20{yr}.{url_ageb_bal[yr][1]}'
    df = pd.read_excel(enebal_de, sheet_name='tj', index_col=[0], usecols=[0, 31], skiprows=list(range(0, 50)),
                       nrows=24, na_values=['-'])
    df.columns = [2000 + yr]
    ht_enduse_de = pd.concat([ht_enduse_de, df], axis=1)
ht_enduse_de = ht_enduse_de / 3.6

# process Austrian energy balances
ht_enduse_at = pd.read_excel(enbal_at, sheet_name='Fernwärme', header=[438], index_col=[0], nrows=24,
                             na_values=['-']).astype('float')
ht_enduse_at = ht_enduse_at / 1000

ht_cons = pd.DataFrame(index=YEARS, columns=pd.MultiIndex.from_product([zones, ['HE08', 'HM08', 'HG08', 'WW', 'IND']]))
ht_cons.loc[YEARS, ('AT', 'HE08')] = ht_enduse_at.loc['Private Haushalte', YEARS] * 0.376 * 0.75
ht_cons.loc[YEARS, ('AT', 'HM08')] = ht_enduse_at.loc['Private Haushalte', YEARS] * 0.624 * 0.75
ht_cons.loc[YEARS, ('AT', 'WW')] = ht_enduse_at.loc['Private Haushalte', YEARS] * 0.25
ht_cons.loc[YEARS, ('AT', 'HG08')] = ht_enduse_at.loc['Öffentliche und Private Dienstleistungen', YEARS]
ht_cons.loc[YEARS, ('AT', 'IND')] = ht_enduse_at.loc['Produzierender Bereich', YEARS]
ht_cons.loc[YEARS, ('DE', 'HE08')] = ht_enduse_de.loc['Haushalte', YEARS] * 0.376 * 0.75
ht_cons.loc[YEARS, ('DE', 'HM08')] = ht_enduse_de.loc['Haushalte', YEARS] * 0.624 * 0.75
ht_cons.loc[YEARS, ('DE', 'WW')] = ht_enduse_de.loc['Haushalte', YEARS] * 0.25
ht_cons.loc[YEARS, ('DE', 'HG08')] = ht_enduse_de.loc['Gewerbe, Handel, Dienstleistungen u.übrige Verbraucher', YEARS]
ht_cons.loc[YEARS, ('DE', 'IND')] = ht_enduse_de.loc['Bergbau, Gew. Steine u. Erden, Verarbeit. Gewerbe insg.', YEARS]

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

cons_pattern = pd.read_excel(CONSUMPTION_PATTERN, 'consumption_pattern', index_col=[0, 1])
cons_pattern = cons_pattern.rename_axis('hour', axis=1)
cons_pattern = cons_pattern.unstack('consumer').stack('hour')

ht_consumption = heat_consumption(zones, YEARS, ht_cons, df_heat, cons_pattern)
ht_consumption.to_csv(heat_cons_file)
logging.info(f'exported hourly heat demand to {heat_cons_file}')
