# SCALERS
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# CLASSIFIERS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# =============================================================================
# Scalers
# =============================================================================
scaler_dict = {'min_max': MinMaxScaler(),
               'z_score': StandardScaler()}
# =============================================================================
# classifiers
# =============================================================================
RF_C = RandomForestClassifier(n_estimators = 20, 
                              max_depth = 2,
                              min_samples_split = 10, 
                              random_state = 0)
LogReg = LogisticRegression()
SVM = SVC()
LDA = LinearDiscriminantAnalysis()
KNN_C = KNeighborsClassifier(n_neighbors = 10)

classifier_dict = {'RandomForestClassifier': RF_C,
                   'SupportVectorMachine': SVM,
                   'LogisticRegression': LogReg, 
                   'LinearDiscriminantAnalysis': LDA,
                   'KNeighborsClassifier': KNN_C}



