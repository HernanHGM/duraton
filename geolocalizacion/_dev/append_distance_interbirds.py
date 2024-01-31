# %% IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

import sys

path = 'E:\\duraton'
if path not in sys.path:
    sys.path.append(path)

# OWN LIBRARIES
from geolocalizacion.distances import add_distances_interbirds
# %% LOAD DATA
path = "E:/duraton/geolocalizacion/_data/fly/interpolated/all_interpolated_data.csv"
# path = "E:/duraton/geolocalizacion/_data/fly/enriquecida_elevation_weather/all_grouped_data.csv"
df = pd.read_csv(path,
                 index_col=False,
                 encoding="ISO-8859-1")


# %% ADD distances
datetime_cols = ['UTC_datetime']
# datetime_cols = ['UTC_date', 'hour', 'flying_situation']
df_enriched = add_distances_interbirds(df, datetime_cols)

# %% SAVE DATA
save_path = "E:/duraton/geolocalizacion/_data/fly/interpolated/all_interpolated_data_distances.csv"
df_enriched.to_csv(save_path, index=False, encoding="ISO-8859-1")
