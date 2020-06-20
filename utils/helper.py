import numpy as np
import pandas as pd

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
            df.drop(col, axis=1, inplace=True)
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
    nan_info.fillna(method='ffill', axis=0, inplace=True)

    # store index of lines containing "unknown" levels
    target_index = []
    for i, row in nan_info.iterrows():
        try:
            if (
                ("unknown" in row.iloc[-1]) or 
                ("not" in row.iloc[-1] and "known" in row.iloc[-1])
            ):
                target_index.append(i)
        except:
            continue
    
    nan_info = nan_info.iloc[target_index, [0,1]]
    nan_info.set_index("Attribute", inplace=True) # index provide attribute, corresponding value is the NaN value for that attribute
    
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

    for row in nan_info.index:
        if row in df.columns:
            if df[row].dtype != 'object':
                nan_info.loc[row,:] = float(nan_info.loc[row,:].values) # make sure the fill_na value type matches corresponding column dtype in original dataframe
                                                     # everything can be set to float except 'object' dtype for which fill_na value which must remain string, already the case
                          
                    
                      
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
    for col in df.columns :
        if col in nan_fill.index:
            try:
                #df.loc[df[col].isna()==True, col] = nan_fill.loc[col, "Value"]
                df.loc[:,col] = df.loc[:,col].replace(nan_fill.loc[col, "Value"], np.nan) # inplace replace is buggy, don't use
            except Exception as e:
                if "Cannot setitem" in str(e):
                    # if no unknown category yet in that column, add the value to the categories
                    df[col].cat.set_categories(np.hstack((df[col].cat.categories.values,
                                     np.nan)), inplace=True)
                    df.loc[:,col] = df.loc[:,col].replace(nan_fill.loc[col, "Value"], np.nan) # inplace replace is buggy, don't use
                else:
                    print(e)
                    
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
    num_info.fillna(method='ffill', axis=0, inplace=True)

    # store index of lines containing "numeric" levels
    target_index = []
    for i, row in num_info.iterrows():
        try:
            if "numeric" in row.iloc[-1]:
                target_index.append(i)
        except:
            continue
    
    num_info = num_info.iloc[target_index, [0,1]]
    num_info.set_index("Attribute", inplace=True) # index provide attribute, corresponding value is the NaN value for that attribute
    

                      
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
            feat1, feat2 = list(str(int(row)))
        elif np.isnan(row):
            feat1, feat2 = [np.nan, np.nan]
        elif isinstance(row, (np.number, float, int)):
            feat1, feat2 = list(str(int(row)))
            
        return [feat1, feat2]
    
    columns = []
    for index, value in df[column].items():
        columns.append((split_content(value)))
    columns = pd.DataFrame(columns, index= df.index, columns=["CAMEO1", "CAMEO2"])
    df = df.join(columns)
    df.drop(column, axis=1, inplace=True)
    
    return df, df.columns