import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
import re

from googletrans import Translator
translator = Translator()


def remove_high_corr(corr, threshold, output_high_corr=False):
    """
    Only  keep one element from a chain of highly (> threshold) correlated elements
    
    Parameters:
    -----------
    corr (pandas.DataFrame) : the correlation matrix
    threshold (float) : the threshold above which a high correlation is considered between elements
    
    Returns:
    --------
    if not output_high_corr:
    (main_elements, correlated_elements) (tuple) : a tuple containing two lists featuring respectively the main elements (to be kept)
                                                   and the correlated elements with the main elements (to be discarded)
    if output_high_corr:
    (-, -, high_corr) : same as above + high_corr which provides a dictionnary providing pairs of highly correlated variables 
    
    """
    
    indices = np.where(corr > threshold)
    high_corr = {corr.index[x] : corr.columns[y] for x, y in zip(*indices)
                                        if x != y and x < y}
    main_elements = []
    correlated_elements = []
    
    for key in high_corr.keys():
        if (key not in main_elements) and (key not in correlated_elements):
            main_elements.append(key)
            correlated_elements.append(high_corr[key])
        elif key in correlated_elements:
            correlated_elements.append(high_corr[key])
        else:
            continue
    
    if not output_high_corr:
        return (main_elements, correlated_elements)
    else:
        return (main_elements, correlated_elements, high_corr)

def drop_full_empty_row(df, df_name=None):
    """
    Drop completely empty rows from  a dataframe
    
    Parameters:
    -----------
    df (pandas.DataFrame) : DataFrame to be processed
    df_name (str) : a human-understandble name of the dataframe for messaging about operations
    
    Returns:
    --------
    None
    """
    if not df_name:
        df_name = "DataFrame"
        
    print(f'Before deletion of full empty rows, we have {len(df)} samples in {df_name} data')
    df.dropna(axis=0, how='all')
    print(f'After deletion of full empty rows, we have {len(df)} samples in {df_name} data')   

def identify_insignificant_columns(df, thresh = 0.8):
    """
    Identify columns where NaN counts / total values exceed thresh
    
    Parameters:
    -----------
    df (pandas.DataFrame) : DataFrame to be processed
    thresh (float) : threshold for deleting columns
    
    Returns:
    --------
    list of insigifnicant columns
    
    """
    removed_col = []
    
    total_count  = df.shape[0]
    
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count/total_count > thresh:
            removed_col.append(col)
     
    return removed_col
    
    
    
def remove_insignificant_columns(df, thresh = 0.8):
    """
    Removes columns where NaN counts / total values exceed thresh
    
    Parameters:
    -----------
    df (pandas.DataFrame) : DataFrame to be processed
    thresh (float) : threshold for deleting columns
    
    Returns:
    --------
    list of removed columns
    
    """
    removed_col = []
    
    total_count  = df.shape[0]
    
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count/total_count > thresh:
            df = df.drop(col, axis=1)
            print(f'column {col} has been dropped')
            removed_col.append(col)
    
    return removed_col

def construct_fill_na(filename, df):
    """
    Construct a dataframe identifying nan values for a dataframe that differ from np.nan
    Construct a dictionnary providing replacements proposals in case multiple equivalent nan.values are identified for same feature
    
    Parameters:
    -----------
    filename (pathlib.Path or str) : path to the filename containing the information about NaN values
                                     ! current function only works on a very specific template for this file
    
    df (pandas.DataFrame) : dataframe for which the nan values must be identified
    
    Returns:
    --------
    nan_info (pd.DataFrame) : dataframe identifying which values in the dataframe df are equivalent to np.nan
    replacements (dict) : a dictionnary whose keys are (some) of the columns of df and whose values are dictionnaries 
                         providing {value_to_be_replaced : value_to_replace}

    
    """    
    
    
    nan_info = pd.read_excel(filename,
                        usecols=[1,3,4])
    nan_info.columns = ["Attribute", "Value", "Meaning"]
    nan_info = nan_info.fillna(method='ffill', axis=0)

    # store index of lines containing "unknown" levels
    target_index = []
    for i, row in nan_info.iterrows():
        try:
            string = row.iloc[-1].split()
            if (
                ("unknown" in string) or 
                ("no" in string and "known" in string)
            ):
                target_index.append(i)
        except:
            continue
    
    nan_info = nan_info.iloc[target_index, [0,1]]
    nan_info = nan_info.set_index("Attribute") # index provide attribute, corresponding value is the NaN value for that attribute
    
    # some attributes have two possible values
    # identify which ones, consider only one value and make
    # the dataframe pop_df & customer_df consistent with the one considered

    replacements = {}
    for i, row in nan_info.iterrows():
        if type(row.values[0]) == str:
            if len(row.values[0].split()) > 1:
                kept, dropped = row.values[0].split()
                nan_info.loc[i] = kept
                replacements[i] = {dropped:kept} # this will be used to replace in original dataframe
    
    # for some replacements, it may happen that a str is read
    # altough the corresponding column in dataframe contains float 
    # so for each column that is dtype numeric, remove anything that could hinder a 
    # conversion from string to float e.g. commas
    for col in df.select_dtypes(include=[np.number]):
        if col in nan_info.index.values:
            if isinstance(nan_info.loc[col,:].values[0], str):
                nan_info.loc[col,:] = float(nan_info.loc[col,:].values[0].replace(',',''))
                if col in replacements.keys():
                    for key, value in replacements[col].items():
                        replacements[col] = {float(key.replace(',','')) : float(value.replace(',',''))}

    # for row in nan_info.index:
    #     if row in df.columns:
    #         if df[row].dtype != 'object':
    #             nan_info.loc[row,:] = float(nan_info.loc[row,:].values) # make sure the fill_na value type matches corresponding column dtype in original dataframe
    #                                                  # everything can be set to float except 'object' dtype for which fill_na value which must remain string, already the case
                          
                    
                      
    return nan_info, replacements


def make_replacement(df, replacements):
    """
    replace values in a dataframe according to a dictionnary

    Parameters:
    -----------
    df (pandas.DataFrame) : the dataframe on which replacements must be made
    replace (dict) : a dictionnary whose keys are (some) of the columns of df and whose values are dictionnaries 
                     providing {value_to_be_replaced : value_to_replace}

    Returns:
    --------
    df (pandas.DataFrame) : modified dataframe with replacements performed

    """
    
    count = 0
    
    for col in df.columns:
        if col in replacements.keys():
            try:
                df.loc[:,col] = df.loc[:,col].replace(replacements[col])
                count+=1
            except Exception as e:
                print(e)
    
    print(f'{count} replacements made')
    
    return df

def fill_na_presc(df, nan_fill):
    """
    replace equivalent NaN identified in nan_fill by np.nan
    
    Parameters:
    -----------
    df (pandas.DataFrame) : Dataframe on which NaN will be replaced
    nan_fill (pandas.DaTaframe) : DataFrame providing the rules for NaN replacement
    
    Returns:
    --------
    None
    
    Remarks:
    --------
    nan_fill shall have its index corresponding to columns in df in which one desires to replace equivalent NaN values
    nan_fill.loc[col, "Value"] shall provide the equivalent NaN value in column col of df to be replaced by np.nan
    ! TODO : make it generic and parameterized

    """
    def replace_combinations(series, replace_value):
        if series.dtype in (np.int32,np.int64, int):
            series = series.replace(int(replace_value), np.nan)
        elif series.dtype in (np.float32,np.float64, float):
            series = series.replace(float(replace_value), np.nan)
        elif series.dtype == 'object':
            # hope a replacement works
            # this is already assumed in construct_fill_na method 
            # since object columns are not handled specifically 
            # it is assumed that the nan equivalent in replacement & nan_fill
            # corresponds to nan value in the dataframe
            series = series.replace(replace_value, np.nan)
        
        return series

    for col in df.columns :
        if col in nan_fill.index:
            try:
                #df.loc[df[col].isna()==True, col] = nan_fill.loc[col, "Value"]
                df.loc[:,col] = replace_combinations(df.loc[:,col], nan_fill.loc[col, "Value"]) # inplace replace is buggy, don't use
            except Exception as e:
                if "Cannot setitem" in str(e):
                    # if no unknown category yet in that column, add the value to the categories
                    df[col].cat = df[col].cat.set_categories(np.hstack((df[col].cat.categories.values,
                                     np.nan)))
                    df.loc[:,col] = replace_combinations(nan_fill.loc[col, "Value"], np.nan) # inplace replace is buggy, don't use
                else:
                    print(e)
     
    return df
                    
def identify_numeric_from_filename(filename, df):
    """
    Identify numeric columns based on information contained in filename
    
    Parameters:
    -----------
    filename (pathlib.Path or str) : path to the filename containing the information about numeric values
                                     ! current function only works on a very specific template for this file
        
    Returns:
    --------
    a list providing the name of numeric columns

    
    """    
    
    
    num_info = pd.read_excel(filename,
                        usecols=[1,3,4])
    num_info.columns = ["Attribute", "Value", "Meaning"]
    num_info = num_info.fillna(method='ffill', axis=0)

    # store index of lines containing "numeric" levels
    target_index = []
    for i, row in num_info.iterrows():
        try:
            if "numeric" in row.iloc[-1]:
                target_index.append(i)
        except:
            continue
    
    num_info = num_info.iloc[target_index, [0,1]]
    num_info = num_info.set_index("Attribute") # index provide attribute, corresponding value is the NaN value for that attribute
    

                      
    return list(num_info.index)


def identify_numeric_from_df(df):
    """
    Identify numeric columns based on df values
    
    Parameters:
    -----------
    df (pandas.DataFrame) : datafrale that will be used to infer numeric columns
        
    Returns:
    --------
    a tuple of 2 lists providing the names of numeric & non-numeric columns

    
    """    
    non_numeric = []
    numeric = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='raise')
            numeric.append(col)
        except Exception as e:
            non_numeric.append(col)
    return numeric, non_numeric


def df_to_numeric(df):
    column_not_converted = []
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except Exception as e:
            print(f'column {col} could not be converted to numeric')
            column_not_converted.append(col)
    return df, column_not_converted


def get_non_numeric(df):
    non_numeric = []
    for col in df.columns:
        if df[col].dtype not in (np.number): #(np.float64, np.int64):
            non_numeric.append(col)
    return non_numeric

def non_frequent_to_nan(df, threshold = 0.02):
    """
    replace non frequent (lower than threshold) occurences inside df with np.nan
    
    Parameters:
    -----------
    df (pandas.DataFrame) : Dataframe in which non-frequent occurences will be replaced
    thresh (float) : threshold for non-frequent categorization

        
    Returns:
    --------
    df with non-frequent occurences replaced with np.nan

    
    """    
    for col in df.columns:
        tot_values = df[col].shape[0]
        value_counts = df[col].value_counts()
        to_remove = value_counts[value_counts/tot_values <= threshold].index
        df[col] = df[col].replace(to_remove, np.nan) # replace inplace buggy
        
        
def split_cameo(df, column):
    """
    split column into two columns with separate information
    
    Parameters:
    -----------
    df (pandas.DataFrame) : Dataframe in which non-frequent occurences will be replaced
    column (str) : name of column to be split

        
    Returns:
    --------
    df with column split

    
    """    
    
    def split_content(row):
        if (isinstance(row,str)) :
            feat1, feat2 = list(str(int(float(row))))
        elif np.isnan(row):
            feat1, feat2 = [np.nan, np.nan]
        elif isinstance(row, (np.number, float, int)):
            feat1, feat2 = list(str(int(row)))
            
        return [feat1, feat2]
    
    print(f'shape before transformation : {df.shape}')
    
    columns = []
    for index, value in df[column].items():
        columns.append((split_content(value)))
    columns = pd.DataFrame(columns, index= df.index, columns=["CAMEO1", "CAMEO2"])
    df = df.join(columns)
    df = df.drop(column, axis=1)
    
    print(f'shape after transformation : {df.shape}')
    
    return df, df.columns

def group_low_freq(series, threshold = 0.05):
    """
    replace low frequency (under threshold) categories with the same low frequency category to reduce categories number (grouping)
    
    Parameters:
    -----------
    series (pandas.Series) : series in which low frequency categories will be grouped under same category
    threshold (float) : threshold below which frequency is deemed low and category will be replaced
    
    Returns:
    --------
    series (pandas.Series) : series with low frequezncy categories grouped under same category
    freq_flag (str) : flag identifiying wheter the processed series met the low frequency ('low') criterion or not ('high')
    
    Notes:
    ------
    if only one low frequency is identified, nothing is changed
    
    """
    
    criterion = 0.05
    freq = series.value_counts()/len(series)
    if (freq<criterion).sum() <= 1:
        freq_flag = 'high'
    else:
        freq_flag = 'low'
        series = series.mask(series.map(freq)<0.05, freq[freq<criterion].index[0]) # where frequency is below criterion
                                                                             # replace with first most frequent value below criterion
    return series,freq_flag


def get_feature_out(estimator, feature_in):
    if hasattr(estimator,'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f'vec_{f}' \
                for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers:
        transformer = ct.named_transformers_[name]
        if name!='remainder':
            if isinstance(transformer, Pipeline):
                current_features = features
                for step in transformer:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(transformer, features)
            output_features.extend(features_out)
        elif estimator=='passthrough':
            output_features.extend(ct._feature_names_in[features])

    return output_features

def find_feature_per_car(feature_df, char):
    """
    Find specific features (attributes) from feature_df based on a characteristic char.
    
    Parameters:
    -----------
    feature_df (pandas.DataFrame) : the dataframe containing 
    the features information. It is expected that one column is named "Attribute" and that another one is named "Data Type"
    
    char (str) : the characteristic that will be searched for
    
    Returns:
    --------
    The name of the features (attributes) that present the characteristic char
    
    """
    
    return set(feature_df.groupby("Attribute")['Data Type'].apply(lambda x: x[x.str.contains(char)]).index.get_level_values(0))

def describe_feature(df, feature):
    """
    describe feature from dataframe df
    
    Parameters:
    -----------
    df (pandas.DataFrame) : the dataframe containing the feature. 
    It is expected that one column is named feature
    
    feature (str) : the feature that needs description
        
    Returns:
    --------
    a dataframe containing the feature description
    
    """
    
    dtype = df[feature].dtype
    
    value_counts = df[feature].value_counts()
    
    n_values = len(value_counts.index)
    
    n_na = df[feature].isna().sum()
    if n_na !=0 :
        percent_na = n_na / len(df[feature])*100
    
    translation = ' '.join([translator.translate(substring, dest='en', src='de').text for substring in re.split('_|\s+',feature)])
    
    indices = []
    values = np.zeros(len(value_counts))
    for i, (index, value) in enumerate(value_counts.iteritems()):
        indices.append(index)
        values[i] = value
    
    print(f'feature {feature} is categorized as {dtype} per panda')
    print(f'It means {translation} in english')
    print(f'it has {n_values} different values')
    
    print('\n'.join(('value {} has {} samples and represents {:2.2%} of data')
                    .format(index, value, percent) for index, value, percent in zip(indices, values, values/len(df[feature]))))
    if n_na != 0:
        print(f'it presents {percent_na:.2f}% natural NaN')
    else:
        print('it has no natural NaN')
        
def process_specific_columns(dataset, avant_list, main_list):
    """
    perform specific processing for specific columns of the dataset
    
    Parameters:
    -----------
    row (pandas.Series) : the row containing the columns to be processed. 
    feature_df (pandas.DataFrame) : the dataframe containing 
    the features information. It is expected that one column is named "Attribute" and that another one is named "Data Type"   
    
    Returns:
    --------
    a row with the processed columns
    
    """
        
    dataset.loc[dataset['PRAEGENDE_JUGENDJAHRE'].isin(avant_list), 'PRAEGENDE_JUGENDJAHRE'] = 1
    dataset.loc[dataset['PRAEGENDE_JUGENDJAHRE'].isin(main_list), 'PRAEGENDE_JUGENDJAHRE'] = 2 # not 0 because corresponds to NaN
    
    # below lines take the two digits numbers and split them into single digit numbers [X, X]
    # nan are converted to [nan, nan]
    dataset.loc[~dataset['CAMEO_INTL_2015'].isna(), 'CAMEO_INTL_2015'] = dataset.loc[~dataset['CAMEO_INTL_2015'].isna(), 'CAMEO_INTL_2015'].astype('int').astype('str').apply(list)
    dataset.loc[dataset['CAMEO_INTL_2015'].isna(), 'CAMEO_INTL_2015'] = dataset.loc[dataset['CAMEO_INTL_2015'].isna(), 'CAMEO_INTL_2015'].apply(lambda x: [np.nan, np.nan])
    cameo_columns = pd.DataFrame(list(dataset['CAMEO_INTL_2015'].values), columns = ['CAMEO1', 'CAMEO2'])
    
    dataset = dataset.join(cameo_columns)
    dataset = dataset.drop('CAMEO_INTL_2015', axis=1)
        
    return dataset    
        
def identify_mainstream(df_features):
    """
    identify mainstream vs avantgarde categories in PRAEGENDE_JUGENDJAHRE column
    
    Parameters:
    -----------
    feature_df (pandas.DataFrame) : the dataframe containing 
    the features information. It is expected that one column is named "Attribute" and that another one is named "Data Type"  
    
    Returns:
    --------
    two lists containing the values of column PRAEGENDE_JUGENDJAHRE that belong to mainstream & avantgarde
    
    """
    
    avant_list = []
    main_list = []
    for index, row in df_features.loc[df_features["Attribute"]=="PRAEGENDE_JUGENDJAHRE",["Value", "Meaning"]].iterrows():
        if 'Avant' in row['Meaning']:
            avant_list.append(row['Value'])
        elif 'Main' in row['Meaning'] : 
            main_list.append(row['Value'])
    
    return avant_list, main_list


def construct_fill_na_new(feature_df, df):
    """
    Construct a dataframe identifying nan values for a dataframe that differ from np.nan
    Construct a dictionnary providing replacements proposals in case multiple equivalent nan.values are identified for same feature
    
    Parameters:
    -----------
    filename (pathlib.Path or str) : path to the filename containing the information about NaN values
                                     ! current function only works on a very specific template for this file
    
    df (pandas.DataFrame) : dataframe for which the nan values must be identified
    
    Returns:
    --------
    nan_info (pd.DataFrame) : dataframe identifying which values in the dataframe df are equivalent to np.nan
    replacements (dict) : a dictionnary whose keys are (some) of the columns of df and whose values are dictionnaries 
                         providing {value_to_be_replaced : value_to_replace}

    
    """    
    
    nan_info = feature_df.copy()

    # store index of lines containing "unknown" levels
    target_index = []
    for i, row in feature_df.iterrows():
        try:
            string = row.loc['Meaning'].split()
            if (
                ("unknown" in string) or 
                ("no" in string and "known" in string)
            ):
                target_index.append(i)
        except:
            continue
    
    nan_info = nan_info.loc[target_index, ['Attribute','Value']]
    nan_info = nan_info.set_index("Attribute") # index provide attribute, corresponding value is the NaN value for that attribute
    
    # some attributes have two possible values
    # identify which ones, consider only one value and make
    # the dataframe pop_df & customer_df consistent with the one considered

    replacements = {}
    for i, row in nan_info.iterrows():
        if type(row.values[0]) == str:
            if len(row.values[0].split()) > 1:
                kept, dropped = row.values[0].split()
                nan_info.loc[i] = kept
                replacements[i] = {dropped:kept} # this will be used to replace in original dataframe
    
    # for some replacements, it may happen that a str is read
    # altough the corresponding column in dataframe contains float 
    # so for each column that is dtype numeric, remove anything that could hinder a 
    # conversion from string to float e.g. commas
    for col in df.select_dtypes(include=[np.number]):
        if col in nan_info.index.values:
            if isinstance(nan_info.loc[col,:].values[0], str):
                nan_info.loc[col,:] = float(nan_info.loc[col,:].values[0].replace(',',''))
                if col in replacements.keys():
                    for key, value in replacements[col].items():
                        replacements[col] = {float(key.replace(',','')) : float(value.replace(',',''))}

    # for row in nan_info.index:
    #     if row in df.columns:
    #         if df[row].dtype != 'object':
    #             nan_info.loc[row,:] = float(nan_info.loc[row,:].values) # make sure the fill_na value type matches corresponding column dtype in original dataframe
    #                                                  # everything can be set to float except 'object' dtype for which fill_na value which must remain string, already the case
                          
                    
                      
    return nan_info, replacements

def drop_na_rows(dataset, thresh):
    """
    Drop rows that contain a fraction of NaN above threshold thresh
    
    Parameters:
    -----------
    dataset (pandas.DataFrame) : dataframe for which the rows will be removed
    thresh (float) : the threshold above which a row gets removed
    
    Returns:
    --------
    the dataset with rows removed according to threshold
       
    """
    print(f'number of rows before dropping : {dataset.shape[0]}')
    dataset = dataset.drop(dataset.loc[dataset.isna().sum(axis=1)/dataset.shape[1]>thresh, :].index , axis=0)
    print(f'number of rows after dropping : {dataset.shape[0]}')
    
    return dataset

def identify_na_columns(dataset, thresh):
    """
    Display columns' names that contain a fraction of NaN above threshold thresh
    
    Parameters:
    -----------
    dataset (pandas.DataFrame) : dataframe for which the columns will be displayed
    thresh (float) : the threshold above which a column gets displayed
    
    Returns:
    --------
    the columns names with a ratio of NaN above threshold
       
    """
    
    columns = dataset.loc[:, dataset.isna().sum()/dataset.shape[0]>thresh].columns
    
    print("The following columns have a ratio of NaN above {:2.0%} : {}".format(thresh, '\n'.join(columns)))
    

def drop_na_columns(dataset, thresh):
    """
    Drop columns that contain a fraction of NaN above threshold thresh
    
    Parameters:
    -----------
    dataset (pandas.DataFrame) : dataframe for which the columns will be removed
    thresh (float) : the threshold above which a column gets removed
    
    Returns:
    --------
    the dataset with columns removed according to threshold
       
    """
    print(f'number of columns before dropping : {dataset.shape[1]}')
    dataset = dataset.drop(dataset.loc[:, dataset.isna().sum()/dataset.shape[0]>thresh].columns , axis=1)
    print(f'number of columns after dropping : {dataset.shape[1]}')
    
    return dataset
    

def identify_categorical_from_analysis(df_features, dataset):
    """
    identify categorical features based on a manual analysis whose insights are contained in df_features
    
    Parameters:
    -----------
    feature_df (pandas.DataFrame) : the dataframe containing 
    the features information. It is expected that one column is named "Attribute" and that another one is named "Data Type"  
    
    dataset (pandas.DataFrame) : the dataset for which the categorical features need to be identified
    
    Returns:
    --------
    the features identified as categorical and which belong to the dataset
    
    """
    
    categorical = set(df_features.loc[df_features['Data Type'].str.contains('Categorical'), :]['Attribute']).intersection(dataset.columns)
    binary = set(df_features.loc[df_features['Data Type'].str.contains('Binary'), :]['Attribute']).intersection(dataset.columns)
    
    return list(categorical.union(binary))





