from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()

setup(
    name='medea-data-atde',
    version='0.1.0',
    author='Sebastian Wehrle',
    author_email='sebastian.wehrle@boku.ac.at',
    description='data processing for the medea power system model',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/sebwehrle/medea-data-atde',
    project_urls={},
    license='MIT',
    packages=find_packages(),
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
        'openpyxl'
    ],
)