# %% imports
import os
import yaml
import logging
import pandas as pd
from config import MEDEA_ROOT_DIR, ERA_DIR, YEARS, zones, imf_file, fx_file, co2_file, url_ageb_sat, url_ageb_bal
from logging_config import setup_logging
from medea_data_atde.fun_get import get_entsoe, download_file, download_era_temp

setup_logging()

# %% Settings

# format for downloading ERA5 temperatures: north/west/south/east
BBOX_CWE = [59.8612, -10.8043, 35.8443, 30.3285]

SERVER = 'sftp-transparency.entsoe.eu'
RAW_DATA_DIR = MEDEA_ROOT_DIR / 'data' / 'raw'

CATEGORIES = [
    'ActualGenerationOutputPerGenerationUnit_16.1.A',
    'AggregatedGenerationPerType_16.1.B_C',
    'AggregatedFillingRateOfWaterReservoirsAndHydroStoragePlants_16.1.D',
    'TotalCommercialSchedules_12.1.F'
    #    'ScheduledCommercialExchanges'
]

credentials = yaml.load(open(MEDEA_ROOT_DIR / 'credentials.yml'), Loader=yaml.SafeLoader)
USER = credentials['entsoe']['user']
PWD = credentials['entsoe']['pwd']


# %% download ENTSO-E data
for cat in CATEGORIES:
    logging.info(f'downloading ENTSO-E transparency data for {cat}')
    get_entsoe(SERVER, USER, PWD, cat, RAW_DATA_DIR)

# %% download era5 temperature (2 m) data
for year in YEARS:
    logging.info(f'downloading ERA5 temperature data for {year}')
    filename = ERA_DIR / f'temperature_europe_{year}.nc'
    download_era_temp(filename, year, BBOX_CWE)

# %% download price data
api_key = credentials['quandl']['apikey']

# IMF commodity price data
# url_imf = 'https://www.imf.org/~/media/Files/Research/CommodityPrices/Monthly/ExternalData.ashx'
url_imf = 'https://www.imf.org/-/media/Files/Research/CommodityPrices/Monthly/external-datadecember.ashx'
logging.info(f'downloading monthly commodity prices from {url_imf}')
download_file(url_imf, imf_file)

# ECB foreign exchange data
url_fx = 'https://sdw.ecb.europa.eu/quickviewexport.do?SERIES_KEY=120.EXR.D.USD.EUR.SP00.A&type=xls'
logging.info(f'downloading exchange rates from {url_fx}')
download_file(url_fx, fx_file)

# CO2 price data
url_co2 = f'https://www.quandl.com/api/v3/datasets/CHRIS/ICE_C1.csv?api_key={api_key}'
logging.info(f'downloading EUA prices from {url_co2}')
download_file(url_co2, co2_file)


# %% download energy balances
# Austrian energy balance as provided by Statistik Austria
url = ('http://www.statistik.at/wcm/idc/idcplg?IdcService=GET_NATIVE_FILE&'
       'RevisionSelectionMethod=LatestReleased&dDocName=029955')
enbal_at = MEDEA_ROOT_DIR / 'data' / 'raw' / 'enbal_AT.xlsx'
logging.info(f'downloading Austrian energy balance')
download_file(url, enbal_at)

# German energy balance as provided by AGEB
for yr in [x - 2000 for x in YEARS]:
    url = 'https://ag-energiebilanzen.de/wp-content/uploads/'
    url_balance = url + f'{url_ageb_bal[yr][0]}/bilanz{yr}d.{url_ageb_bal[yr][1]}'
    url_sat = url + f'{url_ageb_sat[yr][0]}/sat{yr}.{url_ageb_sat[yr][1]}'
    enbal_de = MEDEA_ROOT_DIR / 'data' / 'raw' / f'enbal_DE_20{yr}.{url_ageb_bal[yr][1]}'
    enbal_sat_de = MEDEA_ROOT_DIR / 'data' / 'raw' / f'enbal_sat_DE_20{yr}.{url_ageb_sat[yr][1]}'
    logging.info(f'downloading German energy balance for year 20{yr}')
    download_file(url_balance, enbal_de)
    download_file(url_sat, enbal_sat_de)


ht_cols = pd.MultiIndex.from_product([zones, ['HE08', 'HM08', 'HG08', 'WW', 'IND']])
ht_cons = pd.DataFrame(index=YEARS, columns=ht_cols)


