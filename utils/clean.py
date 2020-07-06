from .helper import *

import io

import s3fs

import boto3
import sagemaker
from sagemaker.amazon.common import write_spmatrix_to_sparse_tensor, write_numpy_to_dense_tensor


import  csv

from googletrans import Translator

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.preprocessing import Normalizer, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

session = sagemaker.session.Session()
role = sagemaker.get_execution_role()
bucket = 'auto-ml-exploration'

INIT_DATA_FOLDER = 'initial_data' 
INITIAL_DATA_SAVEPTH_S3 = f's3://{bucket}/{INIT_DATA_FOLDER}'
s3_dataset_path = f's3://{bucket}/dataset'

CLEANED_DATA_FOLDER = 'cleaned_data' 
CLEANED_DATA_SAVEPTH_S3 = f's3://{bucket}/{CLEANED_DATA_FOLDER}'

TRANSFORMED_DATA_FOLDER = 'transformed_data'
TRANSFORMED_DATA_SAVEPTH_S3 = f's3://{bucket}/{TRANSFORMED_DATA_FOLDER}'


s3fs_handler = s3fs.S3FileSystem()
translator = Translator()

def clean_fn(dataset, flag, fit=True, ct=None):
    """
    Drop rows that contain a fraction of NaN above threshold thresh
    
    Parameters:
    -----------
    dataset (pandas.DataFrame) : dataframe for which the rows will be removed
    fit (Boolean) : wheter the scikit-learn transformers shall be fitted on the data
    
    Returns:
    --------
    the dataset cleaned, scaled, transformed as well as well as the fitted scikit-learn transformers
       
    """
    
    levels_description = pd.read_excel('DIAS Attributes - Values 2017_custom.xlsx', # Added Data Type
                                   header=1, usecols=[1,2,3,4,5,6]).fillna(method = 'ffill')
    features_description = pd.read_excel('DIAS Information Levels - Attributes 2017.xlsx', 
                                         header=1, usecols=[1,2,3,4,5,6]).fillna(method = 'ffill').fillna(method = 'bfill')
    
    global_info = (pd.merge(levels_description, features_description, how='inner', on='Attribute')
               .drop(['Additional notes','Description_y'],axis=1))
    
    # columns to drop based on information gathered from the Excel files
    to_drop = ['CAMEO_DEUG_2015', 'CAMEO_DEU_2015', 'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB',
          'LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 'LP_STATUS_FEIN']
    
    # columns to drop based on detailed analysis of the remaning features not in the Excel files
    to_drop.extend(["ALTER_KIND2", "ALTER_KIND3", "ALTER_KIND4", "ALTER_KIND1",
    "D19_DIGIT_SERV" , "D19_BANKEN_LOKAL", "D19_VERSI_OFFLINE_DATUM", "D19_BANKEN_REST", "D19_VERSI_ONLINE_DATUM", "D19_GARTEN" ,
    "D19_TELKO_ANZ_12", "D19_BANKEN_ANZ_24", "D19_ENERGIE", "D19_VERSI_ANZ_12", "D19_BANKEN_ANZ_12", "D19_BANKEN_GROSS", "D19_BIO_OEKO", 
    "D19_NAHRUNGSERGAENZUNG" , "D19_TELKO_ANZ_24",
    "D19_TELKO_ONLINE_QUOTE_12", "D19_SAMMELARTIKEL", "D19_KOSMETIK", "D19_DROGERIEARTIKEL", "D19_WEIN_FEINKOST", "D19_VERSAND_REST", 
    "D19_TELKO_MOBILE", "D19_TELKO_REST", "D19_VERSI_ANZ_24", "D19_VERSICHERUNGEN", "D19_VERSICHERUNGEN", "D19_VERSI_DATUM", "D19_LEBENSMITTEL", 
    "D19_SCHUHE" , "D19_VERSI_ONLINE_QUOTE_12", "D19_KINDERARTIKEL", "D19_HAUS_DEKO", "D19_BANKEN_DIREKT", "D19_BILDUNG", "D19_RATGEBER", 
    "D19_HANDWERK", "D19_FREIZEIT", "ANZ_KINDER", "D19_LOTTO", "ALTERSKATEGORIE_FEIN", "EINGEZOGENAM_HH_JAHR", "EINGEFUEGT_AM"])
    
    
    
    print(f'number of columns before manual droping : {dataset.shape[1]}')
    dataset = dataset.drop(to_drop, axis=1)
    print(f'number of columns before manual droping : {dataset.shape[1]}')
    
    # first, clean 'X' and 'XX' values that appear and replace them by NaN
    dataset = dataset.replace('X', np.nan)
    dataset = dataset.replace('XX', np.nan)
    
    # then process effectively 
    avant_list, main_list = identify_mainstream(global_info)
    print(f'shape before processing : {dataset.shape}')
    dataset = process_specific_columns(dataset, avant_list, main_list)
    print(f'shape afer processing : {dataset.shape}')
    
    # make non-natural nan values consistent
    nan_info, replacements = construct_fill_na_new(global_info, dataset)
    dataset = make_replacement(dataset, replacements)
    
    # replace non-natural nan by np.nan
    dataset = fill_na_presc(dataset, nan_info)
    
    # print which columns will get removed due to too much NaN
    thresh = .65
    identify_na_columns(dataset, thresh)
    dataset = drop_na_columns(dataset, thresh)
    
    # drop rows due to too much NaN
    # dataset = drop_na_rows(dataset, .05)
    # not performed based on exploratory analysis of mailout training dataset
    
    # save cleaned data to S3
    dataset = dataset.reindex(sorted(dataset.columns), axis=1)
    dataset.to_pickle(f'{CLEANED_DATA_SAVEPTH_S3}/{flag}_cleaned_df.pkl')
    
    # Save index & columns since LNR will be removed for future operations
    # and scikit does not preserve indices
    with open(f"columns_{flag}_cleaned.csv","w") as f:
        wr = csv.writer(f,delimiter="\n")
        wr.writerow(dataset.columns.values)

    with open(f"index_{flag}_cleaned.csv","w") as f:
        wr = csv.writer(f,delimiter="\n")
        wr.writerow(dataset['LNR'].values) # index is contained in LNR columns

    # and upload those to S3 as well
    sagemaker.s3.S3Uploader.upload(f'columns_{flag}_cleaned.csv', 
                                   f'{CLEANED_DATA_SAVEPTH_S3}')

    sagemaker.s3.S3Uploader.upload(f'index_{flag}_cleaned.csv', 
                                   f'{CLEANED_DATA_SAVEPTH_S3}')
    
    # Transform, scale, Input
    
    # First, pop identification column (LNR)
    dataset.drop('LNR', axis=1, inplace=True)
    
    # identify categorical vs numerical
    cat_columns = identify_categorical_from_analysis(global_info, dataset)
    cat_columns = list(set(cat_columns).union(list(dataset.columns[dataset.dtypes == 'object'])))

    num_columns = list(set(dataset.columns).difference(set(cat_columns)))

    print(f'total number of columns:' 
      f'{dataset.shape[1]},\nnumber of categorical:{len(cat_columns)},\n'
      f'number of numerical:{len(num_columns)}')
    
    # define the transformation pipelines
    # define the transformation pipelines
    numeric_pipeline = make_pipeline(SimpleImputer(strategy='mean', missing_values=np.nan), MinMaxScaler())
    categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent', missing_values=np.nan), OneHotEncoder(handle_unknown='ignore'))


    ct = make_column_transformer(
            (numeric_pipeline, num_columns),
            (categorical_pipeline, cat_columns)
                                )

    if fit :
        # fit_transform
        dataset_X = ct.fit_transform(dataset)
    else:
        dataset_X = ct.transform(dataset)
        
    # reconstructing a dataframe
    dataset = pd.DataFrame(dataset_X,columns = get_ct_feature_names(ct))
    dataset['LNR'] = pd.read_csv(f'{CLEANED_DATA_SAVEPTH_S3}/index_{flag}_cleaned.csv', header=None).values
    
    print(f'following imputing, scaling, transforming, dataset has {dataset.shape[1]} features')
    
    # Send transformed data to S3
    dataset = dataset.reindex(sorted(dataset.columns), axis=1)
    dataset.to_pickle(f'{TRANSFORMED_DATA_SAVEPTH_S3}/{flag}_complete_transformed_df.pkl')
    
    # Send transformed data to S3 as recordIO format
    dataset_X = dataset_X.astype('float32', copy=False)

    buf = io.BytesIO()
    #write_spmatrix_to_spaase_tensor(buf, transformed_data) # produces a record IO in fact
    write_numpy_to_dense_tensor(buf, dataset_X)
    buf.seek(0)

    boto3.resource('s3').Bucket(bucket).Object(f'{TRANSFORMED_DATA_SAVEPTH_S3}/{flag}_array').upload_fileobj(buf) 
    
    print(f'recordIO data has been saved to s3://{bucket}/{TRANSFORMED_DATA_SAVEPTH_S3}/{flag}_array')
    
    return dataset, ct