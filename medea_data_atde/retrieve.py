# %% imports
import os
import certifi
import shutil
import pysftp
import urllib3
import cdsapi
import logging
import pandas as pd
from pathlib import Path
from itertools import compress

from config import zones
from logging_config import setup_logging


# ======================================================================================================================
# %% sFTP data download
# ----------------------------------------------------------------------------------------------------------------------
def get_entsoe(connection_string, user, pwd, category, directory):
    """
    downloads dataset from ENTSO-E's transparency data sftp server.
    contact ENTSO-E to receive login credentials.
    :param connection_string: url of ENTSO-E transparency server, as of May 1, 2020: 'sftp-transparency.entsoe.eu'
    :param user: user name required for connecting with sftp server
    :param pwd: password required for connecting with sftp server
    :param category: ENTSO-E data category to be downloaded
    :param directory: directory where downloaded data is saved to. A separate subdirectory is created for each category
    :return: downloaded dataset(s) in dir
    """
    # check if local_dir exists and create if it doesn't
    local_dir = os.path.join(directory, category)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    #
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None
    # connect to entsoe server via sFTP
    entsoe_dir = f'/TP_export/{category}'
    logging.info(f'connecting to {connection_string}')
    with pysftp.Connection(connection_string, username=user, password=pwd, cnopts=cnopts) as sftp:
        sftp.chdir(entsoe_dir)
        files_entsoe = sftp.listdir()
        os.chdir(local_dir)
        files_local = set(os.listdir(local_dir))
        # compare to files on disk
        to_download = list(compress(files_entsoe, [item not in files_local for item in files_entsoe]))
        # download files not on disk
        for file in to_download:
            logging.info(f'starting download of {file}...')
            sftp.get(f'{entsoe_dir}/{file}', os.path.join(directory, category, file))
            logging.info(f'download of {file} successful')

    sftp.close()


def download_file(url, save_to):
    """
    downloads a file from a specified url to disk
    :param url: url-string
    :param save_to: destination file name (string)
    :return:
    """
    http = urllib3.PoolManager(ca_certs=certifi.where())
    with http.request('GET', url, preload_content=False) as r, open(save_to, 'wb') as out_file:
        shutil.copyfileobj(r, out_file)


def download_era_temp(filename, year, bounding_box):
    """
    download daily mean temperatures 2m above surface from ERA5 land data from the copernicus climate data store
    requires registration at https://cds.climate.copernicus.eu/user/register
    for further information see: https://confluence.ecmwf.int/display/CKB/ERA5-Land+data+documentation
    :param filename: path and name of downloaded file
    :param year: year for which daily temperature data is downloaded
    :param bounding_box: bounding box of temperature data
    :return:
    """
    logging.info('downloading bounding box=%s for year=%s', bounding_box, year)
    c = cdsapi.Client()

    if os.path.exists(filename):
        logging.info(f'Skipping {filename}, already exists')
        return

    logging.info(f'starting download of {filename}...')
    for i in range(5):
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': '2m_temperature',
                    'year': f'{year}',
                    'month': [f'{month:02d}' for month in range(1, 13, 1)],
                    'area': bounding_box,
                    'day': [f'{day:02d}' for day in range(1, 32)],
                    'time': [f'{hour:02d}:00' for hour in range(24)],
                },
                f'{filename}.part'
            )
        except Exception as e:
            logging.warning('download failed: %s', e)
        else:
            logging.info(f'download of {filename} successful')
            os.rename(f'{filename}.part', filename)
            break
    else:
        logging.warning('download failed permanently')



def download_energy_balance(country, directory, years=range(2012, 2019)):
    # TODO: check if files exist already and download only if not
    if isinstance(directory, str):
        directory = Path(directory)

    if country == 'AT':
        # Austrian energy balance as provided by Statistik Austria
        url = ('http://www.statistik.at/wcm/idc/idcplg?IdcService=GET_NATIVE_FILE&'
               'RevisionSelectionMethod=LatestReleased&dDocName=029955')
        enbal_at = directory / 'data' / 'raw' / 'enbal_AT.xlsx'
        logging.info(f'downloading Austrian energy balance')
        download_file(url, enbal_at)

    if country == 'DE':
        # German energy balance as provided by AGEB
        url_extension_bal = {12: 'xlsx', 13: 'xls', 14: 'xls', 15: 'xlsx', 16: 'xls', 17: 'xlsx', 18: 'xls', 19: 'xlsx'}
        url_extension_sat = {12: 'xlsx', 13: 'xls', 14: 'xls', 15: 'xlsx', 16: 'xls', 17: 'xlsx', 18: 'xlsx', 19: 'xlsx'}
        for yr in [x - 2000 for x in years]:
            url = 'https://ag-energiebilanzen.de/wp-content/uploads/2021/01/'
            url_balance = url + f'bilanz{yr}d.{url_extension_bal[yr]}'
            url_sat = url + f'sat{yr}.{url_extension_sat[yr]}'
            enbal_de = directory / 'data' / 'raw' / f'enbal_DE_20{yr}.{url_extension_bal[yr]}'
            enbal_sat_de = directory / 'data'/ 'raw' / f'enbal_sat_DE_20{yr}.{url_extension_sat[yr]}'
            logging.info(f'downloading German energy balance for year 20{yr}')
            download_file(url_balance, enbal_de)
            download_file(url_sat, enbal_sat_de)


def is_leapyear(year):
    """
    determines whether a given year is a leap year
    :param year: year to check (numeric)
    :return: boolean
    """
    flag = year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)
    return flag


def days_in_year(year):
    """
    returns number of days in a given year
    :param year: year of interest (numeric)
    :return: number of days in year (numeric)
    """
    if is_leapyear(year):
        return 366
    else:
        return 365


def hours_in_year(year):
    """
    returns number of hours in a goven year
    :param year: year of interest (numeric)
    :return: number of hours in year (numeric)
    """
    if is_leapyear(year):
        return 8784
    else:
        return 8760


def resample_index(index, freq):
    """
    resamples a pandas.DateTimeIndex in daily frequency to
    :param index: pandas.DateTimeIndex to be resampled. Must be daily frequency
    :param freq: pandas frequency string (of higher than daily frequency)
    :return: pandas.DateTimeIndex (resampled)
    """
    assert isinstance(index, pd.DatetimeIndex)
    start_date = index.min()
    end_date = index.max() + pd.DateOffset(days=1)
    resampled_index = pd.date_range(start_date, end_date, freq=freq)[:-1]
    series = pd.Series(resampled_index, resampled_index.floor('D'))
    return pd.DatetimeIndex(series.loc[index].values)


def do_download(medea_root_dir, user, pwd, api_key, years, categories, url_ageb_bal, url_ageb_sat):
    setup_logging()

    # %% Settings
    imf_file = medea_root_dir / 'data' / 'raw' / 'imf_price_data.xlsx'
    fx_file = medea_root_dir / 'data' / 'raw' / 'ecb_fx_data.csv'
    co2_file = medea_root_dir / 'data' / 'raw' / 'eua_price.csv'

    # format for downloading ERA5 temperatures: north/west/south/east
    BBOX_CWE = [59.8612, -10.8043, 35.8443, 30.3285]
    SERVER = 'sftp-transparency.entsoe.eu'
    RAW_DATA_DIR = medea_root_dir / 'data' / 'raw'
    ERA_DIR = medea_root_dir / 'data' / 'raw' / 'era5'

    # % download ENTSO-E data
    for cat in categories:
        logging.info(f'downloading ENTSO-E transparency data for {cat}')
        get_entsoe(SERVER, user, pwd, cat, RAW_DATA_DIR)

    # % download era5 temperature (2 m) data
    for year in years:
        logging.info(f'downloading ERA5 temperature data for {year}')
        filename = ERA_DIR / f'temperature_europe_{year}.nc'
        download_era_temp(filename, year, BBOX_CWE)

    # % download price data
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

    # % download energy balances
    # Austrian energy balance as provided by Statistik Austria
    url = ('http://www.statistik.at/wcm/idc/idcplg?IdcService=GET_NATIVE_FILE&'
           'RevisionSelectionMethod=LatestReleased&dDocName=029955')
    enbal_at = medea_root_dir / 'data' / 'raw' / 'enbal_AT.xlsx'
    logging.info(f'downloading Austrian energy balance')
    download_file(url, enbal_at)

    # German energy balance as provided by AGEB
    for yr in [x - 2000 for x in years]:
        url = 'https://ag-energiebilanzen.de/wp-content/uploads/'
        url_balance = url + f'{url_ageb_bal[yr][0]}/bilanz{yr}d.{url_ageb_bal[yr][1]}'
        url_sat = url + f'{url_ageb_sat[yr][0]}/sat{yr}.{url_ageb_sat[yr][1]}'
        enbal_de = medea_root_dir / 'data' / 'raw' / f'enbal_DE_20{yr}.{url_ageb_bal[yr][1]}'
        enbal_sat_de = medea_root_dir / 'data' / 'raw' / f'enbal_sat_DE_20{yr}.{url_ageb_sat[yr][1]}'
        logging.info(f'downloading German energy balance for year 20{yr}')
        download_file(url_balance, enbal_de)
        download_file(url_sat, enbal_sat_de)

    ht_cols = pd.MultiIndex.from_product([zones, ['HE08', 'HM08', 'HG08', 'WW', 'IND']])
    ht_cons = pd.DataFrame(index=years, columns=ht_cols)
