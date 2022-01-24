# %% imports
from pathlib import Path

# %% global parameters
year = 2016
YEARS = range(2012, 2019, 1)
zones = ['AT', 'DE']
COUNTRY = {'AT': 'Austria', 'DE': 'Germany'}

invest_conventionals = True
invest_renewables = True
invest_storage = True
invest_tc = True

# %% file paths
MEDEA_ROOT_DIR = Path('D:/git_repos/medea-data-atde')
ERA_DIR = MEDEA_ROOT_DIR / 'data' / 'raw' / 'era5'
imf_file = MEDEA_ROOT_DIR / 'data' / 'raw' / 'imf_price_data.xlsx'
fx_file = MEDEA_ROOT_DIR / 'data' / 'raw' / 'ecb_fx_data.csv'
co2_file = MEDEA_ROOT_DIR / 'data' / 'raw' / 'eua_price.csv'

# %% AG Energiebilanzen messy file-dicts
url_ageb_bal = {
    12: ['2021/01', 'xlsx'],
    13: ['2021/01', 'xlsx'],
    14: ['2021/01', 'xls'],
    15: ['2021/01', 'xlsx'],
    16: ['2021/01', 'xls'],
    17: ['2021/01', 'xlsx'],
    18: ['2020/04', 'xlsx'],
    19: ['2021/11', 'xlsx']}

url_ageb_sat = {
    12: ['2021/01', 'xlsx'],
    13: ['2021/01', 'xlsx'],
    14: ['2021/01', 'xls'],
    15: ['2021/01', 'xlsx'],
    16: ['2021/01', 'xls'],
    17: ['2021/01', 'xlsx'],
    18: ['2021/01', 'xlsx'],
    19: ['2021/11', 'xlsx']}
