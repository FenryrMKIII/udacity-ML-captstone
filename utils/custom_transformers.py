from .helper import *

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from pathlib import Path

class naning(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_filepath='attribute.xlsx'): # no *args or **kargs, provides methods get_params() and set_params()
        """
        Parameters:
        -----------
        attribute_filepath (pathlib.Path) : path to the Excel file containing attributes information
        
        """
        
        self.attribute_filepath = attribute_filepath
    
    def fit(self, X, y=None):
                
        if not isinstance(X, pd.DataFrame):
            raise TypeError('only dataframe are handled at the moment')
        
        self.features_names_ = X.columns

                      
        self.nan_info_, self.replacements_ = construct_fill_na(self.attribute_filepath, X) # find nan equivalent
                        
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('only dataframe are handled at the moment')              
        
        make_replacement(X, self.replacements_) # make dataframe consistent if multiple nan equivalent for same feature
        fill_na_presc(X, self.nan_info_) # replace nan equivalent by np.nan

        return X.values


    def get_feature_names(self, *args):
        return self.features_names_ 

    

class cleaning(BaseEstimator, TransformerMixin):
    
    def __init__(self, to_drop = [], ins_threshold=0.6, 
                 corr_threshold=0.7, attribute_filepath='attribute.xlsx'): # no *args or **kargs, provides methods get_params() and set_params()
        """
        Parameters:
        -----------
        to_drop (list) : columns to be dropped
        ins_thresholrd (float) : [0.0 - 1.0] insignificant threshold above which columns containing that proportion of NaN get dropped
        corr_threshold (float) : [0.0 - 1.0] correlation threshold above which correlated columns get dropped (first one is kept)
        attribute_filepath (pathlib.Path) : path to the Excel file containing attributes information
        
        """
        
        self.attribute_filepath = attribute_filepath
        self.ins_threshold = ins_threshold
        self.corr_threshold = corr_threshold
        
        self.to_drop = to_drop
        
    
    def fit(self, X, y=None):
        
        # important to identify columns to be dropped in fit method
        # so that train & test set get identical treatment !
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError('only dataframe are handled at the moment')
        
        col_drop = ['LNR']
        col_kept = X.columns.difference(col_drop, sort = False)
                    
        self.ins_col_ = identify_insignificant_columns(X[col_kept], thresh = self.ins_threshold)
        print(('columns\n{}\nwill be dropped because they' 
               'contain a number of nan above {}%').format('\n'.join(str(x) for x in self.ins_col_), self.ins_threshold*100))
        
        col_drop = col_drop + self.ins_col_
        col_kept = X.columns.difference(col_drop, sort = False)

        corr_ = X[col_kept].corr()
        (main_elements_, self.correlated_col_) = remove_high_corr(corr_, self.corr_threshold)
        print(('columns\n{}\nwill be dropped because they are correlated'
               'above {}% with another one').format('\n'.join(str(x) for x in self.correlated_col_),
                                                   self.corr_threshold*100))

        col_drop = col_drop + self.correlated_col_
        col_kept = X.columns.difference(col_drop, sort = False)
        
        self.object_columns_ = X[col_kept].select_dtypes('object').columns
        print('columns\n{}\n will be considered as object columns'.format('\n'.join(str(x) for x in self.object_columns_)))
              
        self.numeric_, self.non_numeric_ = identify_numeric_from_df(X[col_kept])
        print('columns\n{}\n will be considered as numeric'.format('\n'.join(str(x) for x in self.numeric_)))
        print('columns\n{}\n will be considered as non-numeric'.format('\n'.join(str(x) for x in self.non_numeric_)))
              
        self.nan_info_, self.replacements_ = construct_fill_na(self.attribute_filepath, X[col_kept]) # find nan equivalent
        
        self.features_names_ = col_kept
                
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('only dataframe are handled at the moment')
        
        X.set_index('LNR', inplace=True)
              
        # dropping stuff
        if self.to_drop:
            X.drop(self.to_drop, axis=1, inplace=True)
        drop_full_empty_row(X)
        X.drop(self.ins_col_, axis=1, inplace=True, errors = 'ignore') # ignoring errors since some identified columns could already be dropped in previous steps
        X.drop(self.correlated_col_, axis=1, inplace=True, errors = 'ignore') # ignoring errors since some identified columns could already be dropped in previous steps
              
        # Dealing with object columns
        
        X.loc[:,self.object_columns_] = X.loc[:,self.object_columns_].replace('[X]+', '99942', regex=True) #99942 will be used as a flag for NaN
                                                                                               # mandatory as regex cannot replace by non-string
                                                                                               # inplace not used because not working (apparently, bug according to SO with mixed data)
        
        # replacing nan placeholder immediately
        X = X.replace('99942', np.nan)
#         
        X.loc[:, self.numeric_], _ = df_to_numeric(X.loc[:, self.numeric_])
        X.loc[:, self.non_numeric_] = X.loc[:, self.non_numeric_].astype('category')
              
        
        make_replacement(X, self.replacements_) # make dataframe consistent if multiple nan equivalent for same feature
        fill_na_presc(X, self.nan_info_) # replace nan equivalent by np.nan

        X, self.features_names_ = split_cameo(X, 'CAMEO_INTL_2015')

        return X.values


    def get_feature_names(self, *args):
        return self.features_names_ 
