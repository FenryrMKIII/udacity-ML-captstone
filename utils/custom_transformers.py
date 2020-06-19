from .helper import *

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from pathlib import Path

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
            
        
            
        self.ins_col_ = identify_insignificant_columns(X, thresh = self.ins_threshold)
        print(f'columns {self.ins_col_} will be dropped because they contain a number of nan above {self.ins_threshold*100}%')
        
        corr_ = X.corr()
        (main_elements_, self.correlated_col_) = remove_high_corr(corr_, corr_threshold)
        print(f'''columns {self.correlated_col_} will be dropped because they are correlated /
              above {self.corr_threshold*100}% with another one''')
        
        self.object_columns_ = X.select_dtypes('object').columns
        print(f'columns {self.object_columns_} will be considered as object columns')
              
        self.numeric_, self.non_numeric_ = identify_numeric(X)
        print(f'columns {self.numeric_} will be considered as numeric')
        print(f'columns {self.non_numeric_} will be considered as non-numeric')
              
        self.nan_info_, self.replacements_ = construct_fill_na(self.attribute_filepath, X) # find nan equivalent
        
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('only dataframe are handled at the moment')
        
        X.set_index('LNR', inplace=True)
              
        # dropping stuff
        if self.to_drop:
            X.drop(self.to_drop, axis=1, inplace=True, errors='ignore')
        drop_full_empty_row(X)
        X.drop(self.ins_col_, axis=1, inplace=True)
        X.drop(self.correlated_col_, axis=1, inplace=True)
              
        # Dealing with object columns
        
        X.loc[:,self.object_columns_] = X.loc[:,self.object_columns_].replace('[X]+', '99942', regex=True) #99942 will be used as a flag for NaN
                                                                                               # mandatory as regex cannot replace by non-string
                                                                                               # inplace not used because not working (apparently, bug according to SO with mixed data)
        
        # replacing nan placeholder immediately
        X = X.replace(99942, np.nan)
              
        not_converted_ = df_to_numeric(X)
        X.loc[:,not_converted] = X.loc[:,not_converted_].astype('category')
        X = X.replace(99942, np.nan)
        
        X[self.numeric_] = pd.to_numeric(X[self.numeric], errors='raise')
        X.loc[:,self.non_numeric_] = X.loc[:,self.non_numeric].astype('category')
              
        
        make_replacement(X, self.replacements_) # make dataframe consistent if multiple nan equivalent for same feature
        fill_na_presc(X, self.nan_info_) # replace nan equivalent by np.nan

        make_replacement(X, self.replacements_) # make dataframe consistent if multiple nan equivalent for same feature
        fill_na_presc(X, self.nan_info_) # replace nan equivalent by np.nan
        
        X = split_cameo(X, 'CAMEO_INTL_2015')        
