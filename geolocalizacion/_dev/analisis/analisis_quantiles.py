
# %% BINARIZAMOS VARIABLES X Y CALCULAMOS PERCENTILES
plt.close('all')


print(df['precipMM'].unique())
# =============================================================================
# Creo variables meteorológicas binarizadas 
# =============================================================================
weather_variables_binarized = [variable+'_binarized' for variable in weather_variables]
# Número de intervalos
n_intervalos = 10

for variable in weather_variables:
    # Calcular los intervalos y las frecuencias utilizando pandas.cut
    df['intervalo'] = pd.cut(df[variable], 
                                      bins=n_intervalos, 
                                      include_lowest=True)
    bin_variable = variable+'_binarized'
    # Calcular los valores medios de cada intervalo
    df[bin_variable] = df['intervalo'].apply(lambda x: x.mid).astype('float')

# =============================================================================
# Calculo quantiles para una unica variable de vuelo para cada variable meorológica
# =============================================================================

#Funcion dentro de cuncion para poder pasar los percentiles como parámetro
def custom_q(q):
    return lambda x: x.quantile(q)
quantiles = [0.5, 0.75, 0.9, 0.95]

old_names = [f'<lambda_{i}>' for i, _ in enumerate(quantiles)]
new_names = [f'quantile_{x}' for _, x in enumerate(quantiles)]
rename_dict = dict(zip(old_names, new_names))

dg_concat_fly = pd.DataFrame()

# Calculo los valores para cada variables de vuelo
for fly_variable in fly_variables:
    dg_concat_weather = pd.DataFrame()
    # Calculo los valores para cada variable meteorológica
    for weather_variable in weather_variables_binarized:
        quantile_functions = [custom_q(q) for q in quantiles]
        agg_functions = quantile_functions + ['count']
        # Agrupo por la varibale climatologica y calculo los quantiles
        dg = df.groupby(weather_variable)[fly_variable].agg(agg_functions)
        dg['data_ratio'] = dg['count']/len(df)
        # Renombro las columnas de <lambda-i> al quantil correspondiente
        dg = dg.rename(columns=rename_dict)
        dg['weather_variable'] = weather_variable
        dg = dg.reset_index()
        dg = dg.rename(columns={weather_variable: 'weather_value'})
        
        
        dg_concat_weather = pd.concat([dg_concat_weather, dg])
    
    dg_fly = dg_concat_weather 
    dg_fly['fly_variable'] = fly_variable
    
    dg_concat_fly = pd.concat([dg_concat_fly, dg_fly])
    

# dg_concat_fly = dg_concat_fly.loc[dg_concat_fly['data_ratio']>1/(5*n_intervalos)]
# %% FIT DATA
from scipy.optimize import curve_fit
# define the true objective function
def funcion_potencia(x, a, b, c):
    return a + b*x**c

def funcion_potencia_inversa(x, a, b, c):
    return a + b*x**(-c)

def funcion_exponencial(x, a, b, c):
    return a + b**(c*x)

def funcion_gaussiana(x, base, amplitude, mean, std_dev):
    """
    Calcula el valor de una función gaussiana en un punto dado x.

    Parámetros:
    -----------
    x : float
        Valor de la variable independiente en el que se evalúa la función gaussiana.
    amplitude : float
        Amplitud o altura del pico de la gaussiana.
    mean : float
        Valor medio o centro de la gaussiana.
    std_dev : float
        Desviación estándar de la gaussiana.

    Devuelve:
    ---------
    float
        Valor de la función gaussiana en el punto x.
    """
    return base + amplitude * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)


# weather_variables = ['tempC', 'HeatIndexC', 'DewPointC', 'FeelsLikeC', 'uvIndex',
#                      'WindChillC', 'WindGustKmph', 'windspeedKmph', 'pressure',
#                      'visibility', 'cloudcover', 'precipMM', 'humidity']
# curve fit
fly_variable = 'max_altitude' 
weather_variable = 'precipMM_binarized'
funcion = funcion_exponencial
df_fit = pd.DataFrame()


condition1 = (dg_concat_fly['fly_variable'] == fly_variable)
condition2 = (dg_concat_fly['weather_variable'] == weather_variable)
condition = condition1 & condition2 
df_fit[fly_variable] = dg_concat_fly.loc[condition]['quantile_0.95']
df_fit[weather_variable] = dg_concat_fly.loc[condition]['weather_value']
# df_fit.loc[ = df_fit.assign(flying_time=np.where(dg_fly['flying_time']>3600*24, 
#                                             3600*24, 
#                                             dg_fly['flying_time']))

def fit_data(df, fly_variable, weather_variable, funcion):
    x = df[weather_variable]
    y = df[fly_variable]
    p_opt, p_cov = curve_fit(funcion, x, y)
    # summarize the parameter values
    p_err = np.sqrt(np.diag(p_cov))
    
    p_opt_max = p_opt + p_err
    p_opt_min = p_opt - p_err
    
    x_fit = np.linspace(min(x), max(x), num=1000)
    y_fit = funcion(x_fit, *p_opt)
    y_fit_max = funcion(x_fit, *p_opt_max)
    y_fit_min = funcion(x_fit, *p_opt_min)
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.plot(x_fit, y_fit, 'k-')
    ax.plot(x_fit, y_fit_max, 'r-')
    ax.plot(x_fit, y_fit_min, 'b-')
    df.plot(kind = 'scatter',
            x = weather_variable,
            y = fly_variable,
            color = 'm',
            ax = ax)
    
fit_data(df_fit, fly_variable, weather_variable, funcion_exponencial)
# %% INTERVALUVALUADOS

weather_variables = ['tempC',
                     'windspeedKmph',
                     'cloudcover', 
                     'precipMM']
weather_variables_binarized = [variable+'_binarized' for variable in weather_variables]

n_fly_variables= len(fly_variables)
n_weather_variables = len(weather_variables_binarized)

color_list = ['r', 'm', 'b', 'k', 'g']
quantiles_list = [x for x in reversed(new_names)]

fig, axs = plt.subplots(n_fly_variables,
                        n_weather_variables, 
                        figsize=(18, 10))
plt.subplots_adjust(left = 0.05, right = 0.98,
                    bottom = 0.1, top = 0.95, 
                    wspace = 0.25, hspace = 0.25)
legend_elements = []
for index_fly, fly_variable in enumerate(fly_variables):
    for index_weather, weather_variable in enumerate(weather_variables_binarized):
        condition1 = (dg_concat_fly['fly_variable'] == fly_variable)
        condition2 = (dg_concat_fly['weather_variable'] == weather_variable)
        condition = condition1 & condition2
        df_plot = dg_concat_fly.loc[condition] 
        ax = axs[index_fly, index_weather]     
        for i, quantile in enumerate(quantiles_list):              
            scatter_plot = df_plot.plot(kind='scatter', 
                         x='weather_value', 
                         y=quantile, 
                         color=color_list[i], 
                         ax=ax)#,
                         # label=quantile
            
        ax.set_xlabel(weather_variable)
        ax.set_ylabel(fly_variable)
        
        # Añadir la referencia del scatter plot a la lista de leyendas
        legend_elements.append(scatter_plot)

# Crear la leyenda fuera de los subplots con las referencias de los scatter plots
fig.legend(handles=legend_elements, labels=[f'Quantile {q}' for q in quantiles_list], loc='upper right')

# handles, labels = axs.get_legend_handles_labels()
# fig.legend(loc='outside upper right')
# fig.legend(handles, labels, loc='upper center')
   


