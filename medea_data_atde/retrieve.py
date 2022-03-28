# %% imports
import os
import sysconfig
import certifi
import shutil
import pysftp
import urllib3
import urllib.request
import cdsapi
import logging
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from bs4 import BeautifulSoup
from itertools import compress
from medea_data_atde.logging_config import setup_logging


# ======================================================================================================================
# %% sFTP data download
# ----------------------------------------------------------------------------------------------------------------------
def init_medea_data_atde(root_dir):
    root_dir = Path(root_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(root_dir / 'data'):
        os.makedirs(root_dir / 'data')
    if not os.path.exists(root_dir / 'data' / 'raw'):
        os.makedirs(root_dir / 'data' / 'raw')
    if not os.path.exists(root_dir / 'data' / 'processed'):
        os.makedirs(root_dir / 'data' / 'processed')

    # fetch main gams model
    package_dir = Path(sysconfig.get_path('data'))
    shutil.copyfile(package_dir / 'raw' / 'capacities.csv', root_dir / 'data' / 'raw' / 'capacities.csv')
    shutil.copyfile(package_dir / 'raw' / 'technologies.csv', root_dir / 'data' / 'raw' / 'technologies.csv')
    shutil.copyfile(package_dir / 'raw' / 'operating_region.csv', root_dir / 'data' / 'raw' / 'operating_region.csv')
    shutil.copyfile(package_dir / 'raw' / 'transmission.csv', root_dir / 'data' / 'raw' / 'transmission.csv')
    shutil.copyfile(package_dir / 'raw' / 'external_cost.csv', root_dir / 'data' / 'raw' / 'external_cost.csv')
    shutil.copyfile(package_dir / 'raw' / 'consumption_pattern.csv',
                    root_dir / 'data' / 'raw' / 'consumption_pattern.csv')
    print('>medea data atde< sucessfully initialized')


def get_entsoe(connection_string, user, pwd, category, directory):
    """
    downloads dataset from ENTSO-E's transparency data sftp server.
    contact ENTSO-E to receive login credentials.
    :param connection_string: url of ENTSO-E transparency server, as of May 1, 2020: 'sftp-transparency.entsoe.eu'
    :param user: username required for connecting with sftp server
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
    # connect to entso-e server via sFTP
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


def mastr_gesamtdatenurl(mastr_html):
    """
    retrieves the current download link for the "Gesamtdatenexport" of the German Marktstammdatenregister
    :param mastr_html: url to the page holding the download link for the zipped Gesamtdatenexport-XML
    :return: url to download zipped Gesamtdatenexport XML-file
    """
    html_page = urllib.request.urlopen(mastr_html)
    soup = BeautifulSoup(html_page, 'html.parser')
    for link in soup.findAll('a'):
        mastr_href = link.get('href')
        if (mastr_href is not None) and ('Gesamtdatenexport' in mastr_href) and ('zip' in mastr_href):
            url_mastr = mastr_href
    return url_mastr


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


def download_era_temp(era5_dir, filename, year, bounding_box, cdsurl=None, cdskey=None):
    """
    download daily mean temperatures 2m above surface from ERA5 land data from the copernicus climate data store
    requires registration at https://cds.climate.copernicus.eu/user/register
    for further information see: https://confluence.ecmwf.int/display/CKB/ERA5-Land+data+documentation
    :param era5_dir: Path to directory holding era5 data
    :param cdskey: key for the Copernicus Climate Data Service
    :param cdsurl: url for the Copernicus Climate Data Service
    :param filename: name of downloaded file
    :param year: year for which daily temperature data is downloaded
    :param bounding_box: bounding box of temperature data
    :return:
    """
    # create .cdsapirc if it doesn't exist and insert credentials
    if not os.path.isfile(os.path.expanduser('~') + '/.cdsapirc') and cdsurl is not None and cdskey is not None:
        with open(os.path.expanduser('~') + '/.cdsapirc', 'x') as cdsapirc:
            cdsapirc.write(f'url: {cdsurl} \n')
            cdsapirc.write(f'key: {cdskey}')

    # check if era5 data directory exists and create if it doesn't
    if not os.path.exists(era5_dir):
        os.makedirs(era5_dir)

    filepath = Path(era5_dir) / filename

    logging.info('downloading bounding box=%s for year=%s', bounding_box, year)
    c = cdsapi.Client()

    if os.path.exists(filepath):
        logging.info(f'Skipping {filepath}, already exists')
        return

    logging.info(f'starting download of {filepath}...')
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
                f'{filepath}.part'
            )
        except Exception as e:
            logging.warning('download failed: %s', e)
        else:
            logging.info(f'download of {filepath} successful')
            os.rename(f'{filepath}.part', filepath)
            break
    else:
        logging.warning('download failed permanently')


def download_energy_balance(country, directory, years=range(2012, 2019)):
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
        url_extension_sat = {12: 'xlsx', 13: 'xls', 14: 'xls', 15: 'xlsx', 16: 'xls', 17: 'xlsx', 18: 'xlsx',
                             19: 'xlsx'}
        for yr in [x - 2000 for x in years]:
            url = 'https://ag-energiebilanzen.de/wp-content/uploads/2021/01/'
            url_balance = url + f'bilanz{yr}d.{url_extension_bal[yr]}'
            url_sat = url + f'sat{yr}.{url_extension_sat[yr]}'
            enbal_de = directory / 'data' / 'raw' / f'enbal_DE_20{yr}.{url_extension_bal[yr]}'
            enbal_sat_de = directory / 'data' / 'raw' / f'enbal_sat_DE_20{yr}.{url_extension_sat[yr]}'
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


def do_download(root_dir, zones, user, pwd, api_key, years, categories, url_ageb_bal, url_ageb_sat, cdsurl=None,
                cdskey=None):
    """
    downloads power system and climate data. Intended for use with power system model medea.
    If not existent, it creates the directory structure: root_dir/data/raw
    :param zones: list of ISO 2-digit country codes. Default ['AT', 'DE']
    :param root_dir: path to main directory
    :param user: user name of ENTSO-E's transparency platform
    :param pwd: password for ENTSO-E's transparency platform
    :param api_key: api_key from quandl
    :param years: years for which data should be downloaded
    :param categories: ENTSO-E data descriptors
    :param url_ageb_bal: url to AG Energiebilanzen energy balances
    :param url_ageb_sat: url to AG Energiebilanzen satellite energy balances
    :param cdsurl: url to Copernicus climate data service
    :param cdskey: user key for Copernicus climate data service
    :return:
    """
    setup_logging()
    init_medea_data_atde(root_dir)

    # %% Settings
    # imf_file = root_dir / 'data' / 'raw' / 'imf_price_data.xlsx'
    bafa_file = root_dir / 'data' / 'raw' / 'egas_aufkommen_export_1991.xlsm'
    destatis_file = root_dir / 'data' / 'raw' / 'energiepreisentwicklung_5619001.xlsx'
    eia_file = root_dir / 'data' / 'raw' / 'RBRTEm.xls'
    fx_file = root_dir / 'data' / 'raw' / 'ecb_fx_data.csv'
    co2_file = root_dir / 'data' / 'raw' / 'eua_price.csv'
    mastr_file = root_dir / 'data' / 'raw' / 'mastr.zip'

    # format for downloading ERA5 temperatures: north/west/south/east
    BBOX_CWE = [59.8612, -10.8043, 35.8443, 30.3285]
    SERVER = 'sftp-transparency.entsoe.eu'
    RAW_DATA_DIR = root_dir / 'data' / 'raw'
    ERA_DIR = root_dir / 'data' / 'raw' / 'era5'

    # % download ENTSO-E data
    for cat in categories:
        logging.info(f'downloading ENTSO-E transparency data for {cat}')
        get_entsoe(SERVER, user, pwd, cat, RAW_DATA_DIR)

    # % download era5 temperature (2 m) data
    for year in years:
        logging.info(f'downloading ERA5 temperature data for {year}')
        filename = ERA_DIR / f'temperature_europe_{year}.nc'
        download_era_temp(ERA_DIR, filename, year, BBOX_CWE, cdsurl=cdsurl, cdskey=cdskey)

    # % download price data
    """
    # IMF commodity price data
    # as of February 2022, the IMF does not provide coal prices in USD anymore. 
    # The platform is under maintenance, so coal prices might come back. 
    # Therefore, code for IMF data is temporarily kept.
    f'http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/{database_ID}/{frequency}.{item1 from dimension1}+' \
    f'{item2 from dimension1}+{item N from dimension1}.{item1 from dimension2}+{item2 from dimension2}+' \
    f'{item M from dimension2}?startPeriod = {startdate} & endPeriod = {enddate}'

    database_ID = 'PCPS'
    frequency = 'M'  # monthly, also possible: Q - quarterly, D - daily, A - annual, W - weekly
    item1 = 'POILBRE'  # Brent Crude
    item2 = 'PCOALSA'  # COal South Africa
    item3 = 'PNGASEU'  # Natural Gas European Union

    # url_imf = 'https://www.imf.org/~/media/Files/Research/CommodityPrices/Monthly/ExternalData.ashx'
    url_imf = 'https://www.imf.org/-/media/Files/Research/CommodityPrices/Monthly/external-datajanuary.ashx'
    logging.info(f'downloading monthly commodity prices from {url_imf}')
    download_file(url_imf, imf_file)
    """
    # BAFA data
    url_ngas = 'https://www.bafa.de/SharedDocs/Downloads/DE/Energie/egas_aufkommen_export_1991.xlsm?__blob=publicationFile'
    # destatis coal price index for Germany
    url_coal = 'https://www.destatis.de/DE/Themen/Wirtschaft/Preise/Publikationen/Energiepreise/energiepreisentwicklung-xlsx-5619001.xlsx?__blob=publicationFile'
    # EIA oil prices
    url_brent = 'https://www.eia.gov/dnav/pet/hist_xls/RBRTEm.xls'

    logging.info(f'downloading monthly coal price index from {url_coal}')
    download_file(url_coal, destatis_file)

    logging.info(f'downloading monthly natural gas prices from {url_ngas}')
    download_file(url_ngas, bafa_file)

    logging.info(f'downloading monthly oil prices from {url_brent}')
    download_file(url_brent, eia_file)

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
    enbal_at = root_dir / 'data' / 'raw' / 'enbal_AT.xlsx'
    logging.info(f'downloading Austrian energy balance')
    download_file(url, enbal_at)

    # German energy balance as provided by AGEB
    for yr in [x - 2000 for x in years]:
        url = 'https://ag-energiebilanzen.de/wp-content/uploads/'
        url_balance = url + f'{url_ageb_bal[yr][0]}/bilanz{yr}d.{url_ageb_bal[yr][1]}'
        url_sat = url + f'{url_ageb_sat[yr][0]}/sat{yr}.{url_ageb_sat[yr][1]}'
        enbal_de = root_dir / 'data' / 'raw' / f'enbal_DE_20{yr}.{url_ageb_bal[yr][1]}'
        enbal_sat_de = root_dir / 'data' / 'raw' / f'enbal_sat_DE_20{yr}.{url_ageb_sat[yr][1]}'
        logging.info(f'downloading German energy balance for year 20{yr}')
        download_file(url_balance, enbal_de)
        download_file(url_sat, enbal_sat_de)

    ht_cols = pd.MultiIndex.from_product([zones, ['HE08', 'HM08', 'HG08', 'WW', 'IND']])
    ht_cons = pd.DataFrame(index=years, columns=ht_cols)

    # European power plant fleet (opsd)
    url_opsd = 'https://data.open-power-system-data.org/conventional_power_plants/latest/conventional_power_plants_EU.csv'
    opsd_plant = root_dir / 'data' / 'raw' / 'conventional_power_plants_EU.csv'
    download_file(url_opsd, opsd_plant)

    # time series (opsd)
    url_timeseries = 'https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv'
    opsd_timeseries = root_dir / 'data' / 'raw' / 'time_series_60min_singleindex.csv'
    download_file(url_timeseries, opsd_timeseries)

    # offshore wind power capacities from MaStR
    mastr_html = 'https://www.marktstammdatenregister.de/MaStR/Datendownload'
    download_file(mastr_gesamtdatenurl(mastr_html), mastr_file)
    with ZipFile(mastr_file, 'r') as zippedObject:
        zippedObject.extract('EinheitenWind.xml', root_dir / 'data' / 'raw')
    logging.info('Marktstammdatenregister successfully downloaded and unzipped "EinheitenWind.xml"')

    # e-control Jahresreihen
    url_jahresreihen = 'https://www.e-control.at/documents/1785851/1811609/BStGes-JR1_Bilanz.xlsx'
    econtrol_jahresreihen = root_dir / 'data' / 'raw' / 'BStGes-JR1_Bilanz.xlsx'
    download_file(url_jahresreihen, econtrol_jahresreihen)

    # EE Jahresreihen DE
    url_eejr = 'https://www.erneuerbare-energien.de/EE/Redaktion/DE/Downloads/zeitreihen-zur-entwicklung-der' \
               '-erneuerbaren-energien-in-deutschland-1990-2021-excel-en.xlsx?__blob=publicationFile'
    ee_jahresreihen = root_dir / 'data' / 'raw' / 'zeitreihen-ee-in-de-1990-2021-excel-en.xlsx'
    download_file(url_eejr, ee_jahresreihen)
    logging.info('Data download successfully completed')
