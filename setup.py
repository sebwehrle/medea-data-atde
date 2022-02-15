from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()

setup(
    name='medea_data_atde',
    version='0.1',
    author='Sebastian Wehrle',
    author_email='sebastian.wehrle@boku.ac.at',
    description='data processing for the medea power system model',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/sebwehrle/medea_data_atde',
    project_urls={},
    license='MIT',
    packages=['medea_data_atde'],
    install_requires=[
        'pyyaml',
        'numpy',
        'scipy',
        'pandas',
        'netCDF4',
        'certifi',
        'cdsapi',
        'urllib3',
        'pysftp',
        'openpyxl',
        'xlrd'
    ],
    data_files=[
        ('raw', ['data/raw/capacities.csv']),
        ('raw', ['data/raw/technologies.csv']),
        ('raw', ['data/raw/operating_region.csv']),
        ('raw', ['data/raw/transmission.csv']),
        ('raw', ['data/raw/external_cost.csv']),
    ]
)
