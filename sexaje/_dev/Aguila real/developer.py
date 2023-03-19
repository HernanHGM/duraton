# %% IMPORT LIBRERIAS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import funciones_pajaros.f as f

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# %% CARGO DATOS

file = 'E:\\trabajo_pajaros\\marcajes\\Tabla_2022_09.xls'
df_original = pd.read_excel (file,sheet_name='Hoja1')

# %% LIMPIEZA DATOS


label = ['sexo']
variables = ['peso', 
             'izda L',	
             'izda DV',	
             'dcha L',	
             'dcha DV',	
             'ancho ala', 
             'ala d',	
             'ala v',	
             '7º  1ª',	
             'cañón en 7ª',
             'antebrazo',	
             'cola',	
             'rectrix c',
             'envergadura',
             'longitud total',	
             'long pico',	
             'alto pico',	
             'ancho pico',	
             'long cabeza',	
             'ancho cabeza',	
             'clave']

data = f.individual_selection(df_original, 'Águila real', ['adulto', 'joven', 'subadulto'], label, variables)

data_clean = f.remove_outliers(data, label)
# data_clean = f.drop_nans(data_clean, 0.08)
data_clean = f.promediado_tarsos(data_clean)

# # data_scaled, _, _ = f.scaling_encoding(data_clean, label)
# data_aug = f.feature_augmentation(data_clean)
# # # # Como he multiplicado variables entre ellas, debo volver a escalar
# data_scaled, X_scaled, Y_scaled = f.scaling_encoding(data_aug, label)
cm = data_clean.corr()
# %%
# titulos=['Body mass','Tarsus L width', 'Tarsus DV width','Wing length D','Wing length V','Primary 7',\
#           'Forearm','Tail length','Rectrix','Wingspan','Body length','Bill length',\
#           'Bill height','Bill width','Head length','Head width','Claw']

# magnitudes=['Weigth (g)','Width(mm)', 'Width (mm)','Length (cm)','Length (cm)','Length (cm)',\
#           'Length (cm)','Length (cm)','Length (cm)','Length (cm)','Length (cm)','Length (mm)',\
#           'Length (mm)','Length (mm)','Length (mm)','Length (mm)','Length (mm)']
#     F,p=f_oneway(M[0],M[1],M[2])
#     p_valor = '$p=%.1e$' % (p, ) 

#     ax.text(0.7,1.1, p_valor, fontsize=12,
#             bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2},
#             transform=ax.transAxes) #transform pone x y en funcion de los ejes en vez de como valores abs

from scipy import stats
def boxplot_per_label(df, label_name, nrows, ncols):
    # Get the columns names
    columns = df.columns
    columns = columns.drop(label_name)
    n = len(columns)
    # Create figure and axes 
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8)) 
      
    # Flatten the axes array 
    axes = axes.flatten() 
    fig.set_figheight(15)
    fig.set_figwidth(20)
    
    # Flatten the axes array 
    axes = axes.flatten() 
    # Remove the extra axes 
    for ax in axes[n:]: 
        fig.delaxes(ax) 
    
    # Iterate over the columns in order to make the boxplots
    for i, col in enumerate(columns):
        # Select the axes
        ax = axes[i]
        
        # Create the boxplot
        df.boxplot(col, by=label_name, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('unidades')
        g_data = df.groupby(label_name)[col]
        g_keys = df.groupby(label_name).groups.keys()
        b = []
        for g in g_keys:
            b.append(groups.get_group(g).dropna())
        f_val, p_val = stats.f_oneway(*b)

        textstr = 'P-valor: {:.2e}'.format(p_val)

        plt.text(0.75, 0.95, textstr, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
    # plt.tight_layout() 
    plt.subplots_adjust(top=0.948,
                        bottom=0.04,
                        left=0.042,
                        right=0.992,
                        hspace=0.34,
                        wspace=0.247)
    fig.suptitle('')
    # Show the figure
    plt.show()
    
boxplot_per_label(data_clean, 'sexo', 4, 5)

# %%
def box_plot(data, label): 
  
    # Number of columns 
    n = data.shape[1] 
  
    # Number of rows 
    nrows = n // 2 if n % 2 == 0 else (n // 2) + 1
      
    # Create figure and axes 
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 8)) 
      
    # Flatten the axes array 
    axes = axes.flatten() 
      
    # Remove the extra axes 
    for ax in axes[n:]: 
        fig.delaxes(ax) 
  
    # Plot box plot for each column  
    for i, column in enumerate(data.columns): 
        if column != label: 
            ax = axes[i] 
            data.boxplot(column=column, by=label, ax=ax) 
  
    fig.suptitle("Box Plot", fontsize=18, y=1.02) 
    plt.show() 
  
box_plot(data_clean, 'sexo')
# %% SEX ANALYSIS
var_names = list(data_aug.columns)
for i in var_names[1:]:
    data_clean.boxplot(i, by='sexo')
    # filename = 'E:\\trabajo_pajaros\\marcajes\\ML sexaje\\Aguila real\\graficas\\' + i + '.png'
    # plt.savefig(filename)
# %% RESTO DE TEFLÓN
# linear regression y--> resto teflón X--> longitud total, L8
from sklearn.linear_model import LinearRegression as LR
label = ['resto teflón']
final_var = ['longitud total', 'L media', 'peso', 'cola']
data_def = data_scaled[final_var + label].copy()
X = data_def[final_var].values
y = np.squeeze(data_def[label].values)

model = LR().fit(X,y)
a = model.score(X,y)
b = model.get_params(deep = True)
print(a)
# %% MODELS
# =============================================================================
# Classifiers
# =============================================================================
RF_C = RandomForestClassifier(n_estimators = 20, max_depth = 2, min_samples_split=10, random_state = 0)
LogReg = LogisticRegression()
SVM = SVC()
LDA = LinearDiscriminantAnalysis()
KNN_C = KNeighborsClassifier(n_neighbors = 10)


# %% CALCULO MODELOS
# Se prueban diferentes modelos probando diferente numero de features
# Los features se seleccionan son Sequential forward selection
final_var = ['peso', 'cola', 'L media', 'DV media', 'area tarso']
data_def = data_scaled[label + final_var].copy()
features = data_def.columns[1:]
X = data_def[final_var].values
y = np.squeeze(data_def[label].values)

most_selected_features, kappa_global, accuracy_global = f.best_features(X, y, features, LDA, n_splits = 10, n_features = 4)




# %% SAVE VALUES
n_features = 5
results = np.zeros((n_features, 5), dtype=object)
results[:,0] = most_selected_features
results[:,1:3] = kappa_global
results[:,3:5] = accuracy_global

