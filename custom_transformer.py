from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np


### Custom transformer class for generating the features
class Energy_variables_generator(BaseEstimator, TransformerMixin):
    def __init__(self):
        #print('\n>>>>>>>init() called.\n')
        just_to_add_a_line = 42

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        
        X = X.copy()
        sources = ['Electricity(kBtu)', 'NaturalGas(kBtu)', 'SteamUse(kBtu)']
        
        ### for each row, sorts the amounts of each source of energy then gets the label  
        X['main_energy_source'] =  [row[1].sort_values(ascending = False).index[0] for row 
              in X[sources].iterrows()]

        # Création de variables binaires qui indiquent 
        #pour chacune des sources d'énergie si elle est utilisée ou non
        for source in sources:
            X[source + '_use'] = np.where(X[source] > 0, 1, 0)

            ### float d-type to prevent that these binary features are one-hot encoded during preprocessing
            X[source + '_use'] = X[source + '_use'].astype('float64')
        
        # Elimination des variables 
        X = X.drop(sources, axis = 1)

    
        return X



### Log transformation for features
class log_transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        #print('\n>>>>>>>init() called.\n')
        just_to_add_a_line = 42
    
    def fit(self, X, y = None):
        return self
    
    ### Transforms both target and features
    def transform(self, X, y = None):
        
        X = X.copy()

        X = np.log(X + 1)

    
        return X



