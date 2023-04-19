from setuptools import setup, find_packages

setup(name="duraton",  # Nombre
    version="0.0.0",  # Versión de desarrollo
    description="Library for iberain raptors investigationg projects",  # Descripción del funcionamiento
    author="Hernán García Mayoral",  # Nombre del autor
    author_email='hernangarciamayoral@gmail.com',  # Email del autor
    license="GPL",  # Licencia: MIT, GPL, GPL 2.0...
    # url="https://github.com/adanadmin/",  # Página oficial (si la hay)
    packages = find_packages(
        exclude = ['doc', 'tests', 'tuts', 'temp', 'data', 'setup']),
    # install_requires=[
    #     # SYSTEM
    #     'xlwt == 1.3.0',
    #     'openpyxl == 3.1.2',
    #     # DATA SCIENCE
    #     'pandas == 2.0.0',
    #     'numpy == 1.24.2',
    #     'datetime == 5.1',
    #     # DATA VISUALIZATION
    #     'bokeh == 2.4.3',
    #     'matplotlib == 3.7.1',
    #     # FORECAST
    #     'xgboost == 1.7.5',
    #     'prophet == 1.1.2',
    #     # QANAT
    #     'pyswmm == 1.4.0',
    #     'swmmio == 0.6.2',
    #     'stormreactor == 1.3.0',
    #     'swmm_api == 0.4.16',
        
    # ],
)
