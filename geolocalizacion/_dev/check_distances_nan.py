import pandas as pd
path_agg = "E:\\duraton\\geolocalizacion\\_data\\fly\\interpolated\\all_interpolated_grouped_data.csv"
path_interp = "E:/duraton/geolocalizacion/_data/fly/interpolated/all_interpolated_data_distances.csv"
# %%
df_interp = pd.read_csv(path_interp,
                 index_col=False, 
                 encoding="ISO-8859-1")

df_agg = pd.read_csv(path_agg,
                 index_col=False, 
                 encoding="ISO-8859-1")
# %%

def count_nan_values(df):
    g = df[['name', 
            'km_Conquista', 
            'km_Deleitosa', 
            'km_Gato', 
            'km_Navilla',
            'km_Zorita']]
    nan_counts = g.groupby('name').apply(lambda x: x.isna().sum()).drop('name', axis=1)
    counts = g.groupby('name').size().to_frame('size')
    dg = pd.concat((nan_counts, counts), axis=1)
    
    for column in dg.columns:
        dg[f'nan_percentage_{column}'] = dg[column]/dg['size'] 
    
    dg = dg.drop('nan_percentage_size', axis=1)
    return dg
# %%
aggregated_df = count_nan_values(df_agg)
interpolated_df = count_nan_values(df_interp)

# %% calcula temperatura en funcion de la altura
'''
Esto luego hay que llevarlo a hourly_aggergate.py
'''
gby_columns = ['name']
mean_columns = ['tempC', 'DewPointC', 'windspeedKmph'] 
sum_columns = ['time_step_s', 'distance_2D']

# %% CREATE DICTIONARIES

def positive_function(x):
    suma = x[x>0].sum()
    return suma

def negative_function(x):
    suma = x[x<0].sum()
    return suma
    
mean_dict = {col: 'mean' for col in mean_columns}
sum_dict = {col: 'sum' for col in sum_columns}
altitude_dict = {'distance_height': negative_function}

agg_dict = {}
agg_dict.update(mean_dict)
agg_dict.update(sum_dict)    
agg_dict.update(altitude_dict)    
# %% FIXED GROUPBY
dg = df_interp.groupby(gby_columns)\
    .agg(agg_dict)\
    .rename(columns={'time_step_s': 'flying_time',
                     'distance_2D': 'distance_travelled'})\
    .reset_index()
# df.groupby("Country")
#   .agg({
#    "column1": "sum",
#    "Revenue": my_cool_func,
#    "columnOther": ...
#   })