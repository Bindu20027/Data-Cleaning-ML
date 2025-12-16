from timeit import default_timer as timer
import numpy as np
import pandas as pd
from math import isnan
from sklearn import preprocessing
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from timeit import default_timer as timer
import pandas as pd
import numpy as np
import os
import sys
from loguru import logger
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
# ------------------ UI DESIGN (Background + Styling) ------------------

import streamlit as st

def add_bg_design():
    st.markdown(
        """
        <style>
        /* Background image */
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1504384308090-c894fdcc538d');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        /* Glass card effect for all content */
        .block-container {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.4);
        }

        /* Text styling */
        h1, h2, h3, p, label, span {
            color: white !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: rgba(255, 255, 255, 0.8);
            color: black;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            border: none;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: rgba(255, 255, 255, 1);
            transform: scale(1.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Activate the design
add_bg_design()

class AutoClean:

    def __init__(self, input_data=None, file_path=None, mode='auto', duplicates=False, missing_num=False, missing_categ=False, encode_categ=False, extract_datetime=False, outliers=False, outlier_param=1.5, logfile=True, verbose=False):
        start = timer()
        self._initialize_logger(verbose, logfile)

        original_input_data = None # Store the original data for methods like round_values

        if input_data is not None and isinstance(input_data, pd.DataFrame):
            original_input_data = input_data.copy()
            logger.info('Using provided DataFrame for cleaning.')
        elif file_path is not None:
            logger.info(f'Attempting to load data from file path: {file_path}')
            try:
                # Assuming CSV for now, can add logic for other formats later if needed
                # A more robust solution might infer file type or take 'file_type' as param
                original_input_data = pd.read_csv(file_path)
                logger.info(f'Data loaded successfully from {file_path}.')
            except Exception as e:
                logger.error(f'Failed to load data from {file_path}: {e}')
                raise ValueError(f'Could not load data from file_path: {file_path}. Error: {e}')
        else:
            raise ValueError('Either input_data (DataFrame) or file_path must be provided to AutoClean.')

        # The DataFrame that will be cleaned
        output_data = original_input_data.copy()


        if mode == 'auto':
            duplicates, missing_num, missing_categ, outliers, encode_categ, extract_datetime = 'auto', 'auto', 'auto', 'winz', ['auto'], 's'

        self.mode = mode
        self.duplicates = duplicates
        self.missing_num = missing_num
        self.missing_categ = missing_categ
        self.outliers = outliers
        self.encode_categ = encode_categ
        self.extract_datetime = extract_datetime
        self.outlier_param = outlier_param

        # validate the input parameters - pass the data that will be cleaned
        self._validate_params(output_data, verbose, logfile)

        # initialize our class and start the autoclean process
        self.output = self._clean_data(output_data, original_input_data) # Pass output_data for cleaning, original_input_data for reference

        end = timer()
        logger.info('AutoClean process completed in {} seconds', round(end-start, 6))

        if not verbose:
            print('AutoClean process completed in', round(end-start, 6), 'seconds')
        if logfile:
            print('Logfile saved to:', os.path.join(os.getcwd(), 'autoclean.log'))

    def _initialize_logger(self, verbose, logfile):
        logger.remove()
        if verbose:
            logger.add(sys.stderr, level='INFO')
        if logfile:
            logger.add('autoclean.log', level='DEBUG')

    def _validate_params(self, df, verbose, logfile):
        pass # Placeholder for actual parameter validation logic

    def _clean_data(self, df, original_input_data):
        # Instantiate cleaning modules
        missing_values_handler = MissingValues()
        outliers_handler = Outliers()
        adjust_handler = Adjust()
        encode_categ_handler = EncodeCateg()
        duplicates_handler = Duplicates()

        # Set parameters for each handler based on AutoClean's __init__
        missing_values_handler.missing_num = self.missing_num
        missing_values_handler.missing_categ = self.missing_categ
        outliers_handler.outliers = self.outliers
        outliers_handler.outlier_param = self.outlier_param
        adjust_handler.extract_datetime = self.extract_datetime
        adjust_handler.duplicates = self.duplicates # used by round_values logic check
        adjust_handler.missing_num = self.missing_num
        adjust_handler.missing_categ = self.missing_categ
        adjust_handler.outliers = self.outliers
        adjust_handler.encode_categ = self.encode_categ
        adjust_handler.extract_datetime = self.extract_datetime
        encode_categ_handler.encode_categ = self.encode_categ
        duplicates_handler.duplicates = self.duplicates

        # Apply cleaning steps in a defined order
        df = duplicates_handler.handle(df)
        df = missing_values_handler.handle(df)
        df = outliers_handler.handle(df)
        df = adjust_handler.convert_datetime(df)
        df = encode_categ_handler.handle(df)
        df = adjust_handler.round_values(df, original_input_data)

        return df

class MissingValues:

    def handle(self, df, _n_neighbors=3):
        # function for handling missing values in the data
        if self.missing_num or self.missing_categ:
            logger.info('Started handling of missing values...', str(self.missing_num).upper())
            start = timer()
            self.count_missing = df.isna().sum().sum()

            if self.count_missing != 0:
                logger.info('Found a total of {} missing value(s)', self.count_missing)
                df = df.dropna(how='all')
                df.reset_index(drop=True)

                if self.missing_num: # numeric data
                    logger.info('Started handling of NUMERICAL missing values... Method: "{}"', str(self.missing_num).upper())
                    # automated handling
                    if self.missing_num == 'auto':
                        self.missing_num = 'linreg'
                        lr = LinearRegression()
                        df = MissingValues._lin_regression_impute(self, df, lr)
                        self.missing_num = 'knn'
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='num')
                    # linear regression imputation
                    elif self.missing_num == 'linreg':
                        lr = LinearRegression()
                        df = MissingValues._lin_regression_impute(self, df, lr)
                    # knn imputation
                    elif self.missing_num == 'knn':
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='num')
                    # mean, median or mode imputation
                    elif self.missing_num in ['mean', 'median', 'most_frequent']:
                        imputer = SimpleImputer(strategy=self.missing_num)
                        df = MissingValues._impute(self, df, imputer, type='num')
                    # delete missing values
                    elif self.missing_num == 'delete':
                        df = MissingValues._delete(self, df, type='num')
                        logger.debug('Deletion of {} NUMERIC missing value(s) succeeded', self.count_missing-df.isna().sum().sum())

                if self.missing_categ: # categorical data
                    logger.info('Started handling of CATEGORICAL missing values... Method: "{}"', str(self.missing_categ).upper())
                    # automated handling
                    if self.missing_categ == 'auto':
                        self.missing_categ = 'logreg'
                        lr = LogisticRegression()
                        df = MissingValues._log_regression_impute(self, df, lr)
                        self.missing_categ = 'knn'
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='categ')
                    elif self.missing_categ == 'logreg':
                        lr = LogisticRegression()
                        df = MissingValues._log_regression_impute(self, df, lr)
                    # knn imputation
                    elif self.missing_categ == 'knn':
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='categ')
                    # mode imputation
                    elif self.missing_categ == 'most_frequent':
                        imputer = SimpleImputer(strategy=self.missing_categ)
                        df = MissingValues._impute(self, df, imputer, type='categ')
                    # delete missing values
                    elif self.missing_categ == 'delete':
                        df = MissingValues._delete(self, df, type='categ')
                        logger.debug('Deletion of {} CATEGORICAL missing value(s) succeeded', self.count_missing-df.isna().sum().sum())
            else:
                logger.debug('{} missing values found', self.count_missing)
            end = timer()
            logger.info('Completed handling of missing values in {} seconds', round(end-start, 6))
        else:
            logger.info('Skipped handling of missing values')
        return df

    def _impute(self, df, imputer, type):
        # function for imputing missing values in the data
        cols_num = df.select_dtypes(include=np.number).columns

        if type == 'num':
            # numerical features
            for feature in df.columns:
                if feature in cols_num:
                    if df[feature].isna().sum().sum() != 0:
                        try:
                            df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)))
                            counter = df[feature].isna().sum().sum() - df_imputed.isna().sum().sum()

                            # Assign imputed values directly, let round_values handle final type conversion
                            df[feature] = df_imputed

                            if counter != 0:
                                logger.debug('{} imputation of {} value(s) succeeded for feature "{}"', str(self.missing_num).upper(), counter, feature)
                        except:
                            logger.warning('{} imputation failed for feature "{}"', str(self.missing_num).upper(), feature)
        else:
            # categorical features
            for feature in df.columns:
                if feature not in cols_num:
                    if df[feature].isna().sum()!= 0:
                        try:
                            mapping = dict()
                            mappings = {k: i for i, k in enumerate(df[feature].dropna().unique(), 0)}
                            mapping[feature] = mappings
                            df[feature] = df[feature].map(mapping[feature])

                            df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)), columns=[feature])
                            counter = sum(1 for i, j in zip(list(df_imputed[feature]), list(df[feature])) if i != j)

                            # round to integers before mapping back to original values
                            df[feature] = df_imputed
                            df[feature] = df[feature].round()
                            df[feature] = df[feature].astype('Int64')

                            # map values back to original
                            mappings_inv = {v: k for k, v in mapping[feature].items()}
                            df[feature] = df[feature].map(mappings_inv)
                            if counter != 0:
                                logger.debug('{} imputation of {} value(s) succeeded for feature "{}"', self.missing_categ.upper(), counter, feature)
                        except:
                            logger.warning('{} imputation failed for feature "{}"', str(self.missing_categ).upper(), feature)
        return df

    def _lin_regression_impute(self, df, model):
        # function for predicting missing values with linear regression
        cols_num = df.select_dtypes(include=np.number).columns
        mapping = dict()
        for feature in df.columns:
            if feature not in cols_num:
                # create label mapping for categorical feature values
                mappings = {k: i for i, k in enumerate(df[feature])}
                mapping[feature] = mappings
                df[feature] = df[feature].map(mapping[feature])
        for feature in cols_num:
                try:
                    test_df = df[df[feature].isnull()==True].dropna(subset=[x for x in df.columns if x != feature])
                    train_df = df[df[feature].isnull()==False].dropna(subset=[x for x in df.columns if x != feature])
                    if len(test_df.index) != 0:
                        pipe = make_pipeline(StandardScaler(), model)

                        y = np.log(train_df[feature]) # log-transform the data
                        X_train = train_df.drop(feature, axis=1)
                        test_df.drop(feature, axis=1, inplace=True)

                        try:
                            model = pipe.fit(X_train, y)
                        except:
                            y = train_df[feature] # use non-log-transformed data
                            model = pipe.fit(X_train, y)
                        if (y == train_df[feature]).all():
                            pred = model.predict(test_df)
                        else:
                            pred = np.exp(model.predict(test_df)) # predict values

                        test_df[feature]= pred

                        # Update values directly, let round_values handle final type conversion
                        df[feature].update(test_df[feature])

                        logger.debug('LINREG imputation of {} value(s) succeeded for feature "{}"', len(pred), feature)
                except:
                    logger.warning('LINREG imputation failed for feature "{}"', feature)
        for feature in df.columns:
            try:
                # map categorical feature values back to original
                mappings_inv = {v: k for k, v in mapping[feature].items()}
                df[feature] = df[feature].map(mappings_inv)
            except:
                pass
        return df

    def _log_regression_impute(self, df, model):
        # function for predicting missing values with logistic regression
        cols_num = df.select_dtypes(include=np.number).columns
        mapping = dict()
        for feature in df.columns:
            if feature not in cols_num:
                # create label mapping for categorical feature values
                mappings = {k: i for i, k in enumerate(df[feature])} #.dropna().unique(), 0)}
                mapping[feature] = mappings
                df[feature] = df[feature].map(mapping[feature])

        target_cols = [x for x in df.columns if x not in cols_num]

        for feature in df.columns:
            if feature in target_cols:
                try:
                    test_df = df[df[feature].isnull()==True].dropna(subset=[x for x in df.columns if x != feature])
                    train_df = df[df[feature].isnull()==False].dropna(subset=[x for x in df.columns if x != feature])
                    if len(test_df.index) != 0:
                        pipe = make_pipeline(StandardScaler(), model)

                        y = train_df[feature]
                        train_df.drop(feature, axis=1, inplace=True)
                        test_df.drop(feature, axis=1, inplace=True)

                        model = pipe.fit(train_df, y)

                        pred = model.predict(test_df) # predict values
                        test_df[feature]= pred

                        if (df[feature].fillna(-9999) % 1  == 0).all():
                            # round back to INTs, if original data were INTs
                            test_df[feature] = test_df[feature].round()
                            test_df[feature] = test_df[feature].astype('Int64')
                            df[feature].update(test_df[feature])
                        logger.debug('LOGREG imputation of {} value(s) succeeded for feature "{}"', len(pred), feature)
                except:
                    logger.warning('LOGREG imputation failed for feature "{}"', feature)
        for feature in df.columns:
            try:
                # map categorical feature values back to original
                mappings_inv = {v: k for k, v in mapping[feature].items()}
                df[feature] = df[feature].map(mappings_inv)
            except:
                pass
        return df

    def _delete(self, df, type):
        # function for deleting missing values
        cols_num = df.select_dtypes(include=np.number).columns
        if type == 'num':
            # numerical features
            for feature in df.columns:
                if feature in cols_num:
                    df = df.dropna(subset=[feature])
                    df.reset_index(drop=True)
        else:
            # categorical features
            for feature in df.columns:
                if feature not in cols_num:
                    df = df.dropna(subset=[feature])
                    df.reset_index(drop=True)
        return df

class Outliers:

    def handle(self, df):
        # function for handling of outliers in the data
        if self.outliers:
            logger.info('Started handling of outliers... Method: "{}"', str(self.outliers).upper())
            start = timer()

            if self.outliers in ['auto', 'winz']:
                df = Outliers._winsorization(self, df)
            elif self.outliers == 'delete':
                df = Outliers._delete(self, df)

            end = timer()
            logger.info('Completed handling of outliers in {} seconds', round(end-start, 6))
        else:
            logger.info('Skipped handling of outliers')
        return df

    def _winsorization(self, df):
        # function for outlier winsorization
        cols_num = df.select_dtypes(include=np.number).columns
        for feature in cols_num:
            counter = 0
            # compute outlier bounds
            lower_bound, upper_bound = Outliers._compute_bounds(self, df, feature)
            for row_index, row_val in enumerate(df[feature]):
                if row_val < lower_bound or row_val > upper_bound:
                    if row_val < lower_bound:
                        df.loc[row_index, feature] = lower_bound
                        counter += 1
                    else:
                        df.loc[row_index, feature] = upper_bound
                        counter += 1
            if counter != 0:
                logger.debug('Outlier imputation of {} value(s) succeeded for feature "{}"', counter, feature)
        return df

    def _delete(self, df):
        # function for deleting outliers in the data
        cols_num = df.select_dtypes(include=np.number).columns
        for feature in cols_num:
            counter = 0
            lower_bound, upper_bound = Outliers._compute_bounds(self, df, feature)
            # delete observations containing outliers
            for row_index, row_val in enumerate(df[feature]):
                if row_val < lower_bound or row_val > upper_bound:
                    df = df.drop(row_index)
                    counter +=1
            df = df.reset_index(drop=True)
            if counter != 0:
                logger.debug('Deletion of {} outliers succeeded for feature "{}"', counter, feature)
        return df

    def _compute_bounds(self, df, feature):
        # function that computes the lower and upper bounds for finding outliers in the data
        featureSorted = sorted(df[feature])

        q1, q3 = np.percentile(featureSorted, [25, 75])
        iqr = q3 - q1

        lb = q1 - (self.outlier_param * iqr)
        ub = q3 + (self.outlier_param * iqr)

        return lb, ub

class Adjust:

    def convert_datetime(self, df):
        # function for extracting of datetime values in the data
        if self.extract_datetime:
            logger.info('Started conversion of DATETIME features... Granularity: {}', self.extract_datetime)
            start = timer()
            cols = set(df.columns) ^ set(df.select_dtypes(include=np.number).columns)
            for feature in cols:
                try:
                    # convert features encoded as strings to type datetime ['D','M','Y','h','m','s']
                    df[feature] = pd.to_datetime(df[feature], infer_datetime_format=True)
                    try:
                        df['Day'] = pd.to_datetime(df[feature]).dt.day

                        if self.extract_datetime in ['auto', 'M','Y','h','m','s']:
                            df['Month'] = pd.to_datetime(df[feature]).dt.month

                            if self.extract_datetime in ['auto', 'Y','h','m','s']:
                                df['Year'] = pd.to_datetime(df[feature]).dt.year

                                if self.extract_datetime in ['auto', 'h','m','s']:
                                    df['Hour'] = pd.to_datetime(df[feature]).dt.hour

                                    if self.extract_datetime in ['auto', 'm','s']:
                                        df['Minute'] = pd.to_datetime(df[feature]).dt.minute

                                        if self.extract_datetime in ['auto', 's']:
                                            df['Sec'] = pd.to_datetime(df[feature]).dt.second

                        logger.debug('Conversion to DATETIME succeeded for feature "{}"', feature)

                        try:
                            # check if entries for the extracted dates/times are non-NULL, otherwise drop
                            if (df['Hour'] == 0).all() and (df['Minute'] == 0).all() and (df['Sec'] == 0).all():
                                df.drop('Hour', inplace = True, axis =1 )
                                df.drop('Minute', inplace = True, axis =1 )
                                df.drop('Sec', inplace = True, axis =1 )
                            elif (df['Day'] == 0).all() and (df['Month'] == 0).all() and (df['Year'] == 0).all():
                                df.drop('Day', inplace = True, axis =1 )
                                df.drop('Month', inplace = True, axis =1 )
                                df.drop('Year', inplace = True, axis =1 )
                        except:
                            pass
                    except:
                        # feature cannot be converted to datetime
                        logger.warning('Conversion to DATETIME failed for "{}"', feature)
                except:
                    pass
            end = timer()
            logger.info('Completed conversion of DATETIME features in {} seconds', round(end-start, 4))
        else:
            logger.info('Skipped datetime feature conversion')
        return df

    def round_values(self, df, input_data):
        # function that checks datatypes of features and converts them if necessary
        if self.duplicates or self.missing_num or self.missing_categ or self.outliers or self.encode_categ or self.extract_datetime:
            logger.info('Started feature type conversion...')
            start = timer()
            counter = 0
            cols_num = df.select_dtypes(include=np.number).columns
            for feature in cols_num:
                    # check if all values are integers
                    if (df[feature].fillna(-9999) % 1  == 0).all():
                        try:
                            # encode FLOATs with only 0 as decimals to INT
                            df[feature] = df[feature].astype('Int64')
                            counter += 1
                            logger.debug('Conversion to type INT succeeded for feature "{}"', feature)
                        except:
                            logger.warning('Conversion to type INT failed for feature "{}"', feature)
                    else:
                        try:
                            df[feature] = df[feature].astype(float)
                            # round the number of decimals of FLOATs back to original
                            dec = None
                            for value in input_data[feature]:
                                try:
                                    if dec == None:
                                        dec = str(value)[::-1].find('.')
                                    else:
                                        if str(value)[::-1].find('.') > dec:
                                            dec = str(value)[::-1].find('.')
                                except:
                                    pass
                            df[feature] = df[feature].round(decimals = dec)
                            counter += 1
                            logger.debug('Conversion to type FLOAT succeeded for feature "{}"', feature)
                        except:
                            logger.warning('Conversion to type FLOAT failed for feature "{}"', feature)
            end = timer()
            logger.info('Completed feature type conversion for {} feature(s) in {} seconds', counter, round(end-start, 6))
        else:
            logger.info('Skipped feature type conversion')
        return df

class EncodeCateg:

    def handle(self, df):
        # function for encoding of categorical features in the data
        if self.encode_categ:
            if not isinstance(self.encode_categ, list):
                self.encode_categ = ['auto']
            # select non numeric features
            cols_categ = set(df.columns) ^ set(df.select_dtypes(include=np.number).columns)
            # check if all columns should be encoded
            if len(self.encode_categ) == 1:
                target_cols = cols_categ # encode ALL columns
            else:
                target_cols = self.encode_categ[1] # encode only specific columns
            logger.info('Started encoding categorical features... Method: "{}"', str(self.encode_categ[0]).upper())
            start = timer()
            for feature in target_cols:
                if feature in cols_categ:
                    # columns are column names
                    feature = feature
                else:
                    # columns are indexes
                    feature = df.columns[feature]
                try:
                    # skip encoding of datetime features
                    pd.to_datetime(df[feature])
                    logger.debug('Skipped encoding for DATETIME feature "{}"', feature)
                except:
                    try:
                        if self.encode_categ[0] == 'auto':
                            # ONEHOT encode if not more than 10 unique values to encode
                            if df[feature].nunique() <=10:
                                df = EncodeCateg._to_onehot(self, df, feature)
                                logger.debug('Encoding to ONEHOT succeeded for feature "{}"', feature)
                            # LABEL encode if not more than 20 unique values to encode
                            elif df[feature].nunique() <=20:
                                df = EncodeCateg._to_label(self, df, feature)
                                logger.debug('Encoding to LABEL succeeded for feature "{}"', feature)
                            # skip encoding if more than 20 unique values to encode
                            else:
                                logger.debug('Encoding skipped for feature "{}"', feature)

                        elif self.encode_categ[0] == 'onehot':
                            df = EncodeCateg._to_onehot(self, df, feature)
                            logger.debug('Encoding to {} succeeded for feature "{}"', str(self.encode_categ[0]).upper(), feature)
                        elif self.encode_categ[0] == 'label':
                            df = EncodeCateg._to_label(self, df, feature)
                            logger.debug('Encoding to {} succeeded for feature "{}"', str(self.encode_categ[0]).upper(), feature)
                    except:
                        logger.warning('Encoding to {} failed for feature "{}"', str(self.encode_categ[0]).upper(), feature)
            end = timer()
            logger.info('Completed encoding of categorical features in {} seconds', round(end-start, 6))
        else:
            logger.info('Skipped encoding of categorical features')
        return df

    def _to_onehot(self, df, feature, limit=10):
        # function that encodes categorical features to OneHot encodings
        one_hot = pd.get_dummies(df[feature], prefix=feature)
        if one_hot.shape[1] > limit:
            logger.warning('ONEHOT encoding for feature "{}" creates {} new features. Consider LABEL encoding instead.', feature, one_hot.shape[1])
        # join the encoded df
        df = df.join(one_hot)
        df = df.drop(columns=[feature]) # <--- ADDED: Drop original categorical column
        return df

    def _to_label(self, df, feature):
        # function that encodes categorical features to label encodings
        le = preprocessing.LabelEncoder()

        df[feature + '_lab'] = le.fit_transform(df[feature].values)
        mapping = dict(zip(le.classes_, range(len(le.classes_))))

        for key in mapping:
            try:
                if isnan(key):
                    replace = {mapping[key] : key }
                    df[feature].replace(replace, inplace=True)
            except:
                pass
        df = df.drop(columns=[feature]) # <--- ADDED: Drop original categorical column
        return df

class Duplicates:

    def handle(self, df):
        if self.duplicates:
            logger.info('Started handling of duplicates... Method: "{}"', str(self.duplicates).upper())
            start = timer()
            original = df.shape
            try:
                df.drop_duplicates(inplace=True, ignore_index=False)
                df = df.reset_index(drop=True)
                new = df.shape
                count = original[0] - new[0]
                if count != 0:
                    logger.debug('Deletion of {} duplicate(s) succeeded', count)
                else:
                    logger.debug('{} missing values found', count)
                end = timer()
                logger.info('Completed handling of duplicates in {} seconds', round(end-start, 6))

            except:
                logger.warning('Handling of duplicates failed')
        else:
            logger.info('Skipped handling of duplicates')
        return df





# --- Define RMSE calculation function (if not already in AutoClean scope) ---
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Streamlit UI for File Upload and Parameter Input ---
st.title("AutoClean - Performance-Driven Data Cleaning App")

st.header("1. Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        st.success("Dataset loaded successfully!")
        st.write("Original Data Head:", df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")

if df is not None:
    st.header("2. Configure Cleaning and ML Experiment")

    # Target Column Input
    all_columns = df.columns.tolist()
    target_column_name = st.selectbox(
        "Select your Target Column (y)",
        options=all_columns
    )

    # AutoClean Parameters Input
    st.subheader("AutoClean Strategy Configuration")
    mode_option = st.selectbox("AutoClean Mode", ['auto', 'manual'], index=1) # Default to manual for customization

    autoclean_params = {'mode': mode_option, 'verbose': False}

    if mode_option == 'manual':
        col1, col2 = st.columns(2)
        with col1:
            autoclean_params['missing_num'] = st.selectbox(
                "Handle Numerical Missing Values",
                ['knn', 'mean', 'median', 'most_frequent', 'linreg', 'delete', False],
                index=0 # Default to knn
            )
            autoclean_params['missing_categ'] = st.selectbox(
                "Handle Categorical Missing Values",
                ['most_frequent', 'logreg', 'knn', 'delete', False],
                index=0 # Default to most_frequent
            )
            autoclean_params['duplicates'] = st.selectbox(
                "Handle Duplicates",
                ['auto', False],
                index=0 # Default to auto
            )
        with col2:
            autoclean_params['outliers'] = st.selectbox(
                "Handle Outliers",
                ['winz', 'delete', False],
                index=0 # Default to winz
            )
            autoclean_params['extract_datetime'] = st.selectbox(
                "Extract Datetime Features (Granularity)",
                ['D', 'M', 'Y', 'h', 'm', 's', False],
                index=0 # Default to D
            )
            autoclean_params['encode_categ'] = st.selectbox(
                "Encode Categorical Features",
                [['onehot'], ['label'], ['auto'], False],
                format_func=lambda x: x[0].upper() if isinstance(x, list) else str(x),
                index=0 # Default to onehot
            )
            autoclean_params['outlier_param'] = st.slider("Outlier Multiplier", 0.5, 3.0, 1.5)

    if st.button("Run AutoClean Evaluation"):
        st.write("### Running Evaluation...")

        try:
            y = df[target_column_name]
            X = df.drop(columns=[target_column_name])
        except KeyError:
            st.error(f"Target column '{target_column_name}' not found.")
            st.stop()

        # Split BEFORE cleaning
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Data split: X_train {X_train.shape}, X_val {X_val.shape}")

        # AUTO-DETECT problem type
        if pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 2:
            problem_type = 'regression'
            ml_model = LinearRegression()
            eval_metric = calculate_rmse
            metric_name = 'RMSE'
            st.info("Detected problem type: Regression")
        else:
            problem_type = 'classification'
            ml_model = LogisticRegression(max_iter=1000, solver='liblinear')
            eval_metric = accuracy_score
            metric_name = 'Accuracy'
            st.info("Detected problem type: Classification")

        # Strategies to test
        autoclean_strategies = [autoclean_params, {"mode": "auto", "verbose": False}]

        results = []

        for idx, params in enumerate(autoclean_strategies):
            st.write(f"--- Evaluating Strategy {idx+1}: {params['mode'].upper()} ---")

            # Combine X and y for cleaning to maintain consistent lengths during row deletions
            # Reset index to avoid issues with differing indices after split or prior operations
            train_df_combined = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
            val_df_combined = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)

            # Clean combined TRAIN independently — NO LEAKAGE
            cleaned_train_combined = AutoClean(input_data=train_df_combined.copy(), **params).output.copy()
            
            # Check if target column was removed during cleaning
            if target_column_name not in cleaned_train_combined.columns:
                st.warning(f"Target column '{target_column_name}' was removed during cleaning for training set under strategy {params['mode']}. Skipping model training for this strategy.")
                metric_value = float('nan')
                results.append({"params": params, metric_name: metric_value})
                continue

            # Split back into features and target
            cleaned_train = cleaned_train_combined.drop(columns=[target_column_name])
            y_train_cleaned = cleaned_train_combined[target_column_name]

            # Clean combined VAL independently — NO LEAKAGE
            cleaned_val_combined = AutoClean(input_data=val_df_combined.copy(), **params).output.copy()
            
            # Check if target column was removed during cleaning
            if target_column_name not in cleaned_val_combined.columns:
                st.warning(f"Target column '{target_column_name}' was removed during cleaning for validation set under strategy {params['mode']}. Skipping model training for this strategy.")
                metric_value = float('nan')
                results.append({"params": params, metric_name: metric_value})
                continue

            # Split back into features and target
            cleaned_val = cleaned_val_combined.drop(columns=[target_column_name])
            y_val_cleaned = cleaned_val_combined[target_column_name]

            # Drop non-numeric columns from features
            cleaned_train = cleaned_train.select_dtypes(include=np.number)
            cleaned_val = cleaned_val.select_dtypes(include=np.number)

            # Align columns between train/val (only for numeric features)
            common_cols = list(set(cleaned_train.columns) & set(cleaned_val.columns))
            
            # If no common numeric columns are left, this strategy is not viable
            if not common_cols:
                st.warning(f"No common numeric features found after cleaning for strategy {params['mode'].upper()}. Skipping model training.")
                metric_value = float('nan')
                results.append({"params": params, metric_name: metric_value})
                continue

            cleaned_train = cleaned_train[common_cols]
            cleaned_val = cleaned_val[common_cols]

            # Safety check for feature columns (could be empty if all were dropped)
            if cleaned_train.empty or cleaned_val.empty or cleaned_train.shape[1] == 0 or cleaned_val.shape[1] == 0:
                metric_value = float('nan')
                st.warning(f"No usable features left after cleaning for strategy {params['mode'].upper()}. Skipping model training.")
                results.append({"params": params, metric_name: metric_value})
                continue
            
            # Ensure y_train_cleaned and y_val_cleaned are properly typed for models
            if problem_type == 'classification':
                y_train_cleaned = y_train_cleaned.astype(int)
                y_val_cleaned = y_val_cleaned.astype(int)
            
            # Check if the cleaned target variable became empty due to all rows being dropped
            if y_train_cleaned.empty or y_val_cleaned.empty:
                st.warning(f"Target variable became empty after cleaning for strategy {params['mode'].upper()}. Skipping model training.")
                metric_value = float('nan')
                results.append({"params": params, metric_name: metric_value})
                continue
            
            ml_model.fit(cleaned_train, y_train_cleaned) # Use cleaned y_train
            y_pred = ml_model.predict(cleaned_val)

            # y_val_cleaned and y_pred should now have consistent lengths due to the combined cleaning approach
            metric_value = eval_metric(y_val_cleaned, y_pred) # Use cleaned y_val
            results.append({"params": params, metric_name: metric_value})

        # ----------------------
        # Show Results
        # ----------------------
        st.header("3. Evaluation Results")

        best_val = float('inf') if metric_name == 'RMSE' else -float('inf')
        best_strategy = None

        for r in results:
            st.write(f"Strategy {r['params']} → {metric_name}: {r[metric_name]:.4f}")

            if metric_name == 'RMSE':
                if r[metric_name] < best_val:
                    best_val = r[metric_name]
                    best_strategy = r['params']
            else:
                if r[metric_name] > best_val:
                    best_val = r[metric_name]
                    best_strategy = r['params']

        if best_strategy:
            st.success(f"Best Strategy: {best_strategy}")
            st.success(f"Best {metric_name}: {best_val:.4f}")
        else:
            st.warning("No best strategy found (all strategies might have failed or resulted in NaN metric values).")

        # ----------------------
        # Final Cleaning + Download
        # ----------------------
        st.header("4. Cleaned Data Output")

        # Combine full X and y for final cleaning using the best strategy
        full_df_combined = pd.concat([df.drop(columns=[target_column_name]).copy().reset_index(drop=True), df[target_column_name].reset_index(drop=True)], axis=1)

        if best_strategy:
            final_cleaned_combined = AutoClean(input_data=full_df_combined, **best_strategy).output
            
            if target_column_name not in final_cleaned_combined.columns:
                st.warning(f"Target column '{target_column_name}' was removed during final cleaning with the best strategy. Cannot reconstruct original target. Outputting only cleaned features.")
                final_cleaned_df = final_cleaned_combined # Only features, potentially missing target
            else:
                final_clean_features = final_cleaned_combined.drop(columns=[target_column_name])
                final_clean_target = final_cleaned_combined[target_column_name]
                final_cleaned_df = pd.concat([final_clean_features, final_clean_target], axis=1)
        else:
            st.warning("No best strategy found (perhaps due to errors). Returning original data as cleaned output.")
            final_cleaned_df = df.copy() # Fallback to original if no best strategy

        st.write("Cleaned Data Preview:")
        st.dataframe(final_cleaned_df.head())

        st.download_button(
            label="Download Cleaned Dataset",
            data=final_cleaned_df.to_csv(index=False),
            file_name="cleaned_output.csv",
            mime="text/csv",
        )
                logger.error(f'Failed to load data from {file_path}: {e}')
                raise ValueError(f'Could not load data from file_path: {file_path}. Error: {e}')
        else:
            raise ValueError('Either input_data (DataFrame) or file_path must be provided to AutoClean.')

        # The DataFrame that will be cleaned
        output_data = original_input_data.copy()


        if mode == 'auto':
            duplicates, missing_num, missing_categ, outliers, encode_categ, extract_datetime = 'auto', 'auto', 'auto', 'winz', ['auto'], 's'

        self.mode = mode
        self.duplicates = duplicates
        self.missing_num = missing_num
        self.missing_categ = missing_categ
        self.outliers = outliers
        self.encode_categ = encode_categ
        self.extract_datetime = extract_datetime
        self.outlier_param = outlier_param

        # validate the input parameters - pass the data that will be cleaned
        self._validate_params(output_data, verbose, logfile)

        # initialize our class and start the autoclean process
        self.output = self._clean_data(output_data, original_input_data) # Pass output_data for cleaning, original_input_data for reference

        end = timer()
        logger.info('AutoClean process completed in {} seconds', round(end-start, 6))

        if not verbose:
            print('AutoClean process completed in', round(end-start, 6), 'seconds')
        if logfile:
            print('Logfile saved to:', os.path.join(os.getcwd(), 'autoclean.log'))

    def _initialize_logger(self, verbose, logfile):
        logger.remove()
        if verbose:
            logger.add(sys.stderr, level='INFO')
        if logfile:
            logger.add('autoclean.log', level='DEBUG')

    def _validate_params(self, df, verbose, logfile):
        pass # Placeholder for actual parameter validation logic

    def _clean_data(self, df, original_input_data):
        # Instantiate cleaning modules
        missing_values_handler = MissingValues()
        outliers_handler = Outliers()
        adjust_handler = Adjust()
        encode_categ_handler = EncodeCateg()
        duplicates_handler = Duplicates()

        # Set parameters for each handler based on AutoClean's __init__
        missing_values_handler.missing_num = self.missing_num
        missing_values_handler.missing_categ = self.missing_categ
        outliers_handler.outliers = self.outliers
        outliers_handler.outlier_param = self.outlier_param
        adjust_handler.extract_datetime = self.extract_datetime
        adjust_handler.duplicates = self.duplicates # used by round_values logic check
        adjust_handler.missing_num = self.missing_num
        adjust_handler.missing_categ = self.missing_categ
        adjust_handler.outliers = self.outliers
        adjust_handler.encode_categ = self.encode_categ
        adjust_handler.extract_datetime = self.extract_datetime
        encode_categ_handler.encode_categ = self.encode_categ
        duplicates_handler.duplicates = self.duplicates

        # Apply cleaning steps in a defined order
        df = duplicates_handler.handle(df)
        df = missing_values_handler.handle(df)
        df = outliers_handler.handle(df)
        df = adjust_handler.convert_datetime(df)
        df = encode_categ_handler.handle(df)
        df = adjust_handler.round_values(df, original_input_data)

        return df

class MissingValues:

    def handle(self, df, _n_neighbors=3):
        # function for handling missing values in the data
        if self.missing_num or self.missing_categ:
            logger.info('Started handling of missing values...', str(self.missing_num).upper())
            start = timer()
            self.count_missing = df.isna().sum().sum()

            if self.count_missing != 0:
                logger.info('Found a total of {} missing value(s)', self.count_missing)
                df = df.dropna(how='all')
                df.reset_index(drop=True)

                if self.missing_num: # numeric data
                    logger.info('Started handling of NUMERICAL missing values... Method: "{}"', str(self.missing_num).upper())
                    # automated handling
                    if self.missing_num == 'auto':
                        self.missing_num = 'linreg'
                        lr = LinearRegression()
                        df = MissingValues._lin_regression_impute(self, df, lr)
                        self.missing_num = 'knn'
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='num')
                    # linear regression imputation
                    elif self.missing_num == 'linreg':
                        lr = LinearRegression()
                        df = MissingValues._lin_regression_impute(self, df, lr)
                    # knn imputation
                    elif self.missing_num == 'knn':
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='num')
                    # mean, median or mode imputation
                    elif self.missing_num in ['mean', 'median', 'most_frequent']:
                        imputer = SimpleImputer(strategy=self.missing_num)
                        df = MissingValues._impute(self, df, imputer, type='num')
                    # delete missing values
                    elif self.missing_num == 'delete':
                        df = MissingValues._delete(self, df, type='num')
                        logger.debug('Deletion of {} NUMERIC missing value(s) succeeded', self.count_missing-df.isna().sum().sum())

                if self.missing_categ: # categorical data
                    logger.info('Started handling of CATEGORICAL missing values... Method: "{}"', str(self.missing_categ).upper())
                    # automated handling
                    if self.missing_categ == 'auto':
                        self.missing_categ = 'logreg'
                        lr = LogisticRegression()
                        df = MissingValues._log_regression_impute(self, df, lr)
                        self.missing_categ = 'knn'
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='categ')
                    elif self.missing_categ == 'logreg':
                        lr = LogisticRegression()
                        df = MissingValues._log_regression_impute(self, df, lr)
                    # knn imputation
                    elif self.missing_categ == 'knn':
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='categ')
                    # mode imputation
                    elif self.missing_categ == 'most_frequent':
                        imputer = SimpleImputer(strategy=self.missing_categ)
                        df = MissingValues._impute(self, df, imputer, type='categ')
                    # delete missing values
                    elif self.missing_categ == 'delete':
                        df = MissingValues._delete(self, df, type='categ')
                        logger.debug('Deletion of {} CATEGORICAL missing value(s) succeeded', self.count_missing-df.isna().sum().sum())
            else:
                logger.debug('{} missing values found', self.count_missing)
            end = timer()
            logger.info('Completed handling of missing values in {} seconds', round(end-start, 6))
        else:
            logger.info('Skipped handling of missing values')
        return df

    def _impute(self, df, imputer, type):
        # function for imputing missing values in the data
        cols_num = df.select_dtypes(include=np.number).columns

        if type == 'num':
            # numerical features
            for feature in df.columns:
                if feature in cols_num:
                    if df[feature].isna().sum().sum() != 0:
                        try:
                            df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)))
                            counter = df[feature].isna().sum().sum() - df_imputed.isna().sum().sum()

                            # Assign imputed values directly, let round_values handle final type conversion
                            df[feature] = df_imputed

                            if counter != 0:
                                logger.debug('{} imputation of {} value(s) succeeded for feature "{}"', str(self.missing_num).upper(), counter, feature)
                        except:
                            logger.warning('{} imputation failed for feature "{}"', str(self.missing_num).upper(), feature)
        else:
            # categorical features
            for feature in df.columns:
                if feature not in cols_num:
                    if df[feature].isna().sum()!= 0:
                        try:
                            mapping = dict()
                            mappings = {k: i for i, k in enumerate(df[feature].dropna().unique(), 0)}
                            mapping[feature] = mappings
                            df[feature] = df[feature].map(mapping[feature])

                            df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)), columns=[feature])
                            counter = sum(1 for i, j in zip(list(df_imputed[feature]), list(df[feature])) if i != j)

                            # round to integers before mapping back to original values
                            df[feature] = df_imputed
                            df[feature] = df[feature].round()
                            df[feature] = df[feature].astype('Int64')

                            # map values back to original
                            mappings_inv = {v: k for k, v in mapping[feature].items()}
                            df[feature] = df[feature].map(mappings_inv)
                            if counter != 0:
                                logger.debug('{} imputation of {} value(s) succeeded for feature "{}"', self.missing_categ.upper(), counter, feature)
                        except:
                            logger.warning('{} imputation failed for feature "{}"', str(self.missing_categ).upper(), feature)
        return df

    def _lin_regression_impute(self, df, model):
        # function for predicting missing values with linear regression
        cols_num = df.select_dtypes(include=np.number).columns
        mapping = dict()
        for feature in df.columns:
            if feature not in cols_num:
                # create label mapping for categorical feature values
                mappings = {k: i for i, k in enumerate(df[feature])}
                mapping[feature] = mappings
                df[feature] = df[feature].map(mapping[feature])
        for feature in cols_num:
                try:
                    test_df = df[df[feature].isnull()==True].dropna(subset=[x for x in df.columns if x != feature])
                    train_df = df[df[feature].isnull()==False].dropna(subset=[x for x in df.columns if x != feature])
                    if len(test_df.index) != 0:
                        pipe = make_pipeline(StandardScaler(), model)

                        y = np.log(train_df[feature]) # log-transform the data
                        X_train = train_df.drop(feature, axis=1)
                        test_df.drop(feature, axis=1, inplace=True)

                        try:
                            model = pipe.fit(X_train, y)
                        except:
                            y = train_df[feature] # use non-log-transformed data
                            model = pipe.fit(X_train, y)
                        if (y == train_df[feature]).all():
                            pred = model.predict(test_df)
                        else:
                            pred = np.exp(model.predict(test_df)) # predict values

                        test_df[feature]= pred

                        # Update values directly, let round_values handle final type conversion
                        df[feature].update(test_df[feature])

                        logger.debug('LINREG imputation of {} value(s) succeeded for feature "{}"', len(pred), feature)
                except:
                    logger.warning('LINREG imputation failed for feature "{}"', feature)
        for feature in df.columns:
            try:
                # map categorical feature values back to original
                mappings_inv = {v: k for k, v in mapping[feature].items()}
                df[feature] = df[feature].map(mappings_inv)
            except:
                pass
        return df

    def _log_regression_impute(self, df, model):
        # function for predicting missing values with logistic regression
        cols_num = df.select_dtypes(include=np.number).columns
        mapping = dict()
        for feature in df.columns:
            if feature not in cols_num:
                # create label mapping for categorical feature values
                mappings = {k: i for i, k in enumerate(df[feature])} #.dropna().unique(), 0)}
                mapping[feature] = mappings
                df[feature] = df[feature].map(mapping[feature])

        target_cols = [x for x in df.columns if x not in cols_num]

        for feature in df.columns:
            if feature in target_cols:
                try:
                    test_df = df[df[feature].isnull()==True].dropna(subset=[x for x in df.columns if x != feature])
                    train_df = df[df[feature].isnull()==False].dropna(subset=[x for x in df.columns if x != feature])
                    if len(test_df.index) != 0:
                        pipe = make_pipeline(StandardScaler(), model)

                        y = train_df[feature]
                        train_df.drop(feature, axis=1, inplace=True)
                        test_df.drop(feature, axis=1, inplace=True)

                        model = pipe.fit(train_df, y)

                        pred = model.predict(test_df) # predict values
                        test_df[feature]= pred

                        if (df[feature].fillna(-9999) % 1  == 0).all():
                            # round back to INTs, if original data were INTs
                            test_df[feature] = test_df[feature].round()
                            test_df[feature] = test_df[feature].astype('Int64')
                            df[feature].update(test_df[feature])
                        logger.debug('LOGREG imputation of {} value(s) succeeded for feature "{}"', len(pred), feature)
                except:
                    logger.warning('LOGREG imputation failed for feature "{}"', feature)
        for feature in df.columns:
            try:
                # map categorical feature values back to original
                mappings_inv = {v: k for k, v in mapping[feature].items()}
                df[feature] = df[feature].map(mappings_inv)
            except:
                pass
        return df

    def _delete(self, df, type):
        # function for deleting missing values
        cols_num = df.select_dtypes(include=np.number).columns
        if type == 'num':
            # numerical features
            for feature in df.columns:
                if feature in cols_num:
                    df = df.dropna(subset=[feature])
                    df.reset_index(drop=True)
        else:
            # categorical features
            for feature in df.columns:
                if feature not in cols_num:
                    df = df.dropna(subset=[feature])
                    df.reset_index(drop=True)
        return df

class Outliers:

    def handle(self, df):
        # function for handling of outliers in the data
        if self.outliers:
            logger.info('Started handling of outliers... Method: "{}"', str(self.outliers).upper())
            start = timer()

            if self.outliers in ['auto', 'winz']:
                df = Outliers._winsorization(self, df)
            elif self.outliers == 'delete':
                df = Outliers._delete(self, df)

            end = timer()
            logger.info('Completed handling of outliers in {} seconds', round(end-start, 6))
        else:
            logger.info('Skipped handling of outliers')
        return df

    def _winsorization(self, df):
        # function for outlier winsorization
        cols_num = df.select_dtypes(include=np.number).columns
        for feature in cols_num:
            counter = 0
            # compute outlier bounds
            lower_bound, upper_bound = Outliers._compute_bounds(self, df, feature)
            for row_index, row_val in enumerate(df[feature]):
                if row_val < lower_bound or row_val > upper_bound:
                    if row_val < lower_bound:
                        df.loc[row_index, feature] = lower_bound
                        counter += 1
                    else:
                        df.loc[row_index, feature] = upper_bound
                        counter += 1
            if counter != 0:
                logger.debug('Outlier imputation of {} value(s) succeeded for feature "{}"', counter, feature)
        return df

    def _delete(self, df):
        # function for deleting outliers in the data
        cols_num = df.select_dtypes(include=np.number).columns
        for feature in cols_num:
            counter = 0
            lower_bound, upper_bound = Outliers._compute_bounds(self, df, feature)
            # delete observations containing outliers
            for row_index, row_val in enumerate(df[feature]):
                if row_val < lower_bound or row_val > upper_bound:
                    df = df.drop(row_index)
                    counter +=1
            df = df.reset_index(drop=True)
            if counter != 0:
                logger.debug('Deletion of {} outliers succeeded for feature "{}"', counter, feature)
        return df

    def _compute_bounds(self, df, feature):
        # function that computes the lower and upper bounds for finding outliers in the data
        featureSorted = sorted(df[feature])

        q1, q3 = np.percentile(featureSorted, [25, 75])
        iqr = q3 - q1

        lb = q1 - (self.outlier_param * iqr)
        ub = q3 + (self.outlier_param * iqr)

        return lb, ub

class Adjust:

    def convert_datetime(self, df):
        # function for extracting of datetime values in the data
        if self.extract_datetime:
            logger.info('Started conversion of DATETIME features... Granularity: {}', self.extract_datetime)
            start = timer()
            cols = set(df.columns) ^ set(df.select_dtypes(include=np.number).columns)
            for feature in cols:
                try:
                    # convert features encoded as strings to type datetime ['D','M','Y','h','m','s']
                    df[feature] = pd.to_datetime(df[feature], infer_datetime_format=True)
                    try:
                        df['Day'] = pd.to_datetime(df[feature]).dt.day

                        if self.extract_datetime in ['auto', 'M','Y','h','m','s']:
                            df['Month'] = pd.to_datetime(df[feature]).dt.month

                            if self.extract_datetime in ['auto', 'Y','h','m','s']:
                                df['Year'] = pd.to_datetime(df[feature]).dt.year

                                if self.extract_datetime in ['auto', 'h','m','s']:
                                    df['Hour'] = pd.to_datetime(df[feature]).dt.hour

                                    if self.extract_datetime in ['auto', 'm','s']:
                                        df['Minute'] = pd.to_datetime(df[feature]).dt.minute

                                        if self.extract_datetime in ['auto', 's']:
                                            df['Sec'] = pd.to_datetime(df[feature]).dt.second

                        logger.debug('Conversion to DATETIME succeeded for feature "{}"', feature)

                        try:
                            # check if entries for the extracted dates/times are non-NULL, otherwise drop
                            if (df['Hour'] == 0).all() and (df['Minute'] == 0).all() and (df['Sec'] == 0).all():
                                df.drop('Hour', inplace = True, axis =1 )
                                df.drop('Minute', inplace = True, axis =1 )
                                df.drop('Sec', inplace = True, axis =1 )
                            elif (df['Day'] == 0).all() and (df['Month'] == 0).all() and (df['Year'] == 0).all():
                                df.drop('Day', inplace = True, axis =1 )
                                df.drop('Month', inplace = True, axis =1 )
                                df.drop('Year', inplace = True, axis =1 )
                        except:
                            pass
                    except:
                        # feature cannot be converted to datetime
                        logger.warning('Conversion to DATETIME failed for "{}"', feature)
                except:
                    pass
            end = timer()
            logger.info('Completed conversion of DATETIME features in {} seconds', round(end-start, 4))
        else:
            logger.info('Skipped datetime feature conversion')
        return df

    def round_values(self, df, input_data):
        # function that checks datatypes of features and converts them if necessary
        if self.duplicates or self.missing_num or self.missing_categ or self.outliers or self.encode_categ or self.extract_datetime:
            logger.info('Started feature type conversion...')
            start = timer()
            counter = 0
            cols_num = df.select_dtypes(include=np.number).columns
            for feature in cols_num:
                    # check if all values are integers
                    if (df[feature].fillna(-9999) % 1  == 0).all():
                        try:
                            # encode FLOATs with only 0 as decimals to INT
                            df[feature] = df[feature].astype('Int64')
                            counter += 1
                            logger.debug('Conversion to type INT succeeded for feature "{}"', feature)
                        except:
                            logger.warning('Conversion to type INT failed for feature "{}"', feature)
                    else:
                        try:
                            df[feature] = df[feature].astype(float)
                            # round the number of decimals of FLOATs back to original
                            dec = None
                            for value in input_data[feature]:
                                try:
                                    if dec == None:
                                        dec = str(value)[::-1].find('.')
                                    else:
                                        if str(value)[::-1].find('.') > dec:
                                            dec = str(value)[::-1].find('.')
                                except:
                                    pass
                            df[feature] = df[feature].round(decimals = dec)
                            counter += 1
                            logger.debug('Conversion to type FLOAT succeeded for feature "{}"', feature)
                        except:
                            logger.warning('Conversion to type FLOAT failed for feature "{}"', feature)
            end = timer()
            logger.info('Completed feature type conversion for {} feature(s) in {} seconds', counter, round(end-start, 6))
        else:
            logger.info('Skipped feature type conversion')
        return df

class EncodeCateg:

    def handle(self, df):
        # function for encoding of categorical features in the data
        if self.encode_categ:
            if not isinstance(self.encode_categ, list):
                self.encode_categ = ['auto']
            # select non numeric features
            cols_categ = set(df.columns) ^ set(df.select_dtypes(include=np.number).columns)
            # check if all columns should be encoded
            if len(self.encode_categ) == 1:
                target_cols = cols_categ # encode ALL columns
            else:
                target_cols = self.encode_categ[1] # encode only specific columns
            logger.info('Started encoding categorical features... Method: "{}"', str(self.encode_categ[0]).upper())
            start = timer()
            for feature in target_cols:
                if feature in cols_categ:
                    # columns are column names
                    feature = feature
                else:
                    # columns are indexes
                    feature = df.columns[feature]
                try:
                    # skip encoding of datetime features
                    pd.to_datetime(df[feature])
                    logger.debug('Skipped encoding for DATETIME feature "{}"', feature)
                except:
                    try:
                        if self.encode_categ[0] == 'auto':
                            # ONEHOT encode if not more than 10 unique values to encode
                            if df[feature].nunique() <=10:
                                df = EncodeCateg._to_onehot(self, df, feature)
                                logger.debug('Encoding to ONEHOT succeeded for feature "{}"', feature)
                            # LABEL encode if not more than 20 unique values to encode
                            elif df[feature].nunique() <=20:
                                df = EncodeCateg._to_label(self, df, feature)
                                logger.debug('Encoding to LABEL succeeded for feature "{}"', feature)
                            # skip encoding if more than 20 unique values to encode
                            else:
                                logger.debug('Encoding skipped for feature "{}"', feature)

                        elif self.encode_categ[0] == 'onehot':
                            df = EncodeCateg._to_onehot(self, df, feature)
                            logger.debug('Encoding to {} succeeded for feature "{}"', str(self.encode_categ[0]).upper(), feature)
                        elif self.encode_categ[0] == 'label':
                            df = EncodeCateg._to_label(self, df, feature)
                            logger.debug('Encoding to {} succeeded for feature "{}"', str(self.encode_categ[0]).upper(), feature)
                    except:
                        logger.warning('Encoding to {} failed for feature "{}"', str(self.encode_categ[0]).upper(), feature)
            end = timer()
            logger.info('Completed encoding of categorical features in {} seconds', round(end-start, 6))
        else:
            logger.info('Skipped encoding of categorical features')
        return df

    def _to_onehot(self, df, feature, limit=10):
        # function that encodes categorical features to OneHot encodings
        one_hot = pd.get_dummies(df[feature], prefix=feature)
        if one_hot.shape[1] > limit:
            logger.warning('ONEHOT encoding for feature "{}" creates {} new features. Consider LABEL encoding instead.', feature, one_hot.shape[1])
        # join the encoded df
        df = df.join(one_hot)
        df = df.drop(columns=[feature]) # <--- ADDED: Drop original categorical column
        return df

    def _to_label(self, df, feature):
        # function that encodes categorical features to label encodings
        le = preprocessing.LabelEncoder()

        df[feature + '_lab'] = le.fit_transform(df[feature].values)
        mapping = dict(zip(le.classes_, range(len(le.classes_))))

        for key in mapping:
            try:
                if isnan(key):
                    replace = {mapping[key] : key }
                    df[feature].replace(replace, inplace=True)
            except:
                pass
        df = df.drop(columns=[feature]) # <--- ADDED: Drop original categorical column
        return df

class Duplicates:

    def handle(self, df):
        if self.duplicates:
            logger.info('Started handling of duplicates... Method: "{}"', str(self.duplicates).upper())
            start = timer()
            original = df.shape
            try:
                df.drop_duplicates(inplace=True, ignore_index=False)
                df = df.reset_index(drop=True)
                new = df.shape
                count = original[0] - new[0]
                if count != 0:
                    logger.debug('Deletion of {} duplicate(s) succeeded', count)
                else:
                    logger.debug('{} missing values found', count)
                end = timer()
                logger.info('Completed handling of duplicates in {} seconds', round(end-start, 6))

            except:
                logger.warning('Handling of duplicates failed')
        else:
            logger.info('Skipped handling of duplicates')
        return df





# --- Define RMSE calculation function (if not already in AutoClean scope) ---
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Streamlit UI for File Upload and Parameter Input ---
st.title("AutoClean - Performance-Driven Data Cleaning App")

st.header("1. Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        st.success("Dataset loaded successfully!")
        st.write("Original Data Head:", df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")

if df is not None:
    st.header("2. Configure Cleaning and ML Experiment")

    # Target Column Input
    all_columns = df.columns.tolist()
    target_column_name = st.selectbox(
        "Select your Target Column (y)",
        options=all_columns
    )

    # AutoClean Parameters Input
    st.subheader("AutoClean Strategy Configuration")
    mode_option = st.selectbox("AutoClean Mode", ['auto', 'manual'], index=1) # Default to manual for customization

    autoclean_params = {'mode': mode_option, 'verbose': False}

    if mode_option == 'manual':
        col1, col2 = st.columns(2)
        with col1:
            autoclean_params['missing_num'] = st.selectbox(
                "Handle Numerical Missing Values",
                ['knn', 'mean', 'median', 'most_frequent', 'linreg', 'delete', False],
                index=0 # Default to knn
            )
            autoclean_params['missing_categ'] = st.selectbox(
                "Handle Categorical Missing Values",
                ['most_frequent', 'logreg', 'knn', 'delete', False],
                index=0 # Default to most_frequent
            )
            autoclean_params['duplicates'] = st.selectbox(
                "Handle Duplicates",
                ['auto', False],
                index=0 # Default to auto
            )
        with col2:
            autoclean_params['outliers'] = st.selectbox(
                "Handle Outliers",
                ['winz', 'delete', False],
                index=0 # Default to winz
            )
            autoclean_params['extract_datetime'] = st.selectbox(
                "Extract Datetime Features (Granularity)",
                ['D', 'M', 'Y', 'h', 'm', 's', False],
                index=0 # Default to D
            )
            autoclean_params['encode_categ'] = st.selectbox(
                "Encode Categorical Features",
                [['onehot'], ['label'], ['auto'], False],
                format_func=lambda x: x[0].upper() if isinstance(x, list) else str(x),
                index=0 # Default to onehot
            )
            autoclean_params['outlier_param'] = st.slider("Outlier Multiplier", 0.5, 3.0, 1.5)

    if st.button("Run AutoClean Evaluation"):
        st.write("### Running Evaluation...")

        try:
            y = df[target_column_name]
            X = df.drop(columns=[target_column_name])
        except KeyError:
            st.error(f"Target column '{target_column_name}' not found.")
            st.stop()

        # Split BEFORE cleaning
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Data split: X_train {X_train.shape}, X_val {X_val.shape}")

        # AUTO-DETECT problem type
        if pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 2:
            problem_type = 'regression'
            ml_model = LinearRegression()
            eval_metric = calculate_rmse
            metric_name = 'RMSE'
            st.info("Detected problem type: Regression")
        else:
            problem_type = 'classification'
            ml_model = LogisticRegression(max_iter=1000, solver='liblinear')
            eval_metric = accuracy_score
            metric_name = 'Accuracy'
            st.info("Detected problem type: Classification")

        # Strategies to test
        autoclean_strategies = [autoclean_params, {"mode": "auto", "verbose": False}]

        results = []

        for idx, params in enumerate(autoclean_strategies):
            st.write(f"--- Evaluating Strategy {idx+1}: {params['mode'].upper()} ---")

            # Combine X and y for cleaning to maintain consistent lengths during row deletions
            # Reset index to avoid issues with differing indices after split or prior operations
            train_df_combined = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
            val_df_combined = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)

            # Clean combined TRAIN independently — NO LEAKAGE
            cleaned_train_combined = AutoClean(input_data=train_df_combined.copy(), **params).output.copy()
            
            # Check if target column was removed during cleaning
            if target_column_name not in cleaned_train_combined.columns:
                st.warning(f"Target column '{target_column_name}' was removed during cleaning for training set under strategy {params['mode']}. Skipping model training for this strategy.")
                metric_value = float('nan')
                results.append({"params": params, metric_name: metric_value})
                continue

            # Split back into features and target
            cleaned_train = cleaned_train_combined.drop(columns=[target_column_name])
            y_train_cleaned = cleaned_train_combined[target_column_name]

            # Clean combined VAL independently — NO LEAKAGE
            cleaned_val_combined = AutoClean(input_data=val_df_combined.copy(), **params).output.copy()
            
            # Check if target column was removed during cleaning
            if target_column_name not in cleaned_val_combined.columns:
                st.warning(f"Target column '{target_column_name}' was removed during cleaning for validation set under strategy {params['mode']}. Skipping model training for this strategy.")
                metric_value = float('nan')
                results.append({"params": params, metric_name: metric_value})
                continue

            # Split back into features and target
            cleaned_val = cleaned_val_combined.drop(columns=[target_column_name])
            y_val_cleaned = cleaned_val_combined[target_column_name]

            # Drop non-numeric columns from features
            cleaned_train = cleaned_train.select_dtypes(include=np.number)
            cleaned_val = cleaned_val.select_dtypes(include=np.number)

            # Align columns between train/val (only for numeric features)
            common_cols = list(set(cleaned_train.columns) & set(cleaned_val.columns))
            
            # If no common numeric columns are left, this strategy is not viable
            if not common_cols:
                st.warning(f"No common numeric features found after cleaning for strategy {params['mode'].upper()}. Skipping model training.")
                metric_value = float('nan')
                results.append({"params": params, metric_name: metric_value})
                continue

            cleaned_train = cleaned_train[common_cols]
            cleaned_val = cleaned_val[common_cols]

            # Safety check for feature columns (could be empty if all were dropped)
            if cleaned_train.empty or cleaned_val.empty or cleaned_train.shape[1] == 0 or cleaned_val.shape[1] == 0:
                metric_value = float('nan')
                st.warning(f"No usable features left after cleaning for strategy {params['mode'].upper()}. Skipping model training.")
                results.append({"params": params, metric_name: metric_value})
                continue
            
            # Ensure y_train_cleaned and y_val_cleaned are properly typed for models
            if problem_type == 'classification':
                y_train_cleaned = y_train_cleaned.astype(int)
                y_val_cleaned = y_val_cleaned.astype(int)
            
            # Check if the cleaned target variable became empty due to all rows being dropped
            if y_train_cleaned.empty or y_val_cleaned.empty:
                st.warning(f"Target variable became empty after cleaning for strategy {params['mode'].upper()}. Skipping model training.")
                metric_value = float('nan')
                results.append({"params": params, metric_name: metric_value})
                continue
            
            ml_model.fit(cleaned_train, y_train_cleaned) # Use cleaned y_train
            y_pred = ml_model.predict(cleaned_val)

            # y_val_cleaned and y_pred should now have consistent lengths due to the combined cleaning approach
            metric_value = eval_metric(y_val_cleaned, y_pred) # Use cleaned y_val
            results.append({"params": params, metric_name: metric_value})

        # ----------------------
        # Show Results
        # ----------------------
        st.header("3. Evaluation Results")

        best_val = float('inf') if metric_name == 'RMSE' else -float('inf')
        best_strategy = None

        for r in results:
            st.write(f"Strategy {r['params']} → {metric_name}: {r[metric_name]:.4f}")

            if metric_name == 'RMSE':
                if r[metric_name] < best_val:
                    best_val = r[metric_name]
                    best_strategy = r['params']
            else:
                if r[metric_name] > best_val:
                    best_val = r[metric_name]
                    best_strategy = r['params']

        if best_strategy:
            st.success(f"Best Strategy: {best_strategy}")
            st.success(f"Best {metric_name}: {best_val:.4f}")
        else:
            st.warning("No best strategy found (all strategies might have failed or resulted in NaN metric values).")

        # ----------------------
        # Final Cleaning + Download
        # ----------------------
        st.header("4. Cleaned Data Output")

        # Combine full X and y for final cleaning using the best strategy
        full_df_combined = pd.concat([df.drop(columns=[target_column_name]).copy().reset_index(drop=True), df[target_column_name].reset_index(drop=True)], axis=1)

        if best_strategy:
            final_cleaned_combined = AutoClean(input_data=full_df_combined, **best_strategy).output
            
            if target_column_name not in final_cleaned_combined.columns:
                st.warning(f"Target column '{target_column_name}' was removed during final cleaning with the best strategy. Cannot reconstruct original target. Outputting only cleaned features.")
                final_cleaned_df = final_cleaned_combined # Only features, potentially missing target
            else:
                final_clean_features = final_cleaned_combined.drop(columns=[target_column_name])
                final_clean_target = final_cleaned_combined[target_column_name]
                final_cleaned_df = pd.concat([final_clean_features, final_clean_target], axis=1)
        else:
            st.warning("No best strategy found (perhaps due to errors). Returning original data as cleaned output.")
            final_cleaned_df = df.copy() # Fallback to original if no best strategy

        st.write("Cleaned Data Preview:")
        st.dataframe(final_cleaned_df.head())

        st.download_button(
            label="Download Cleaned Dataset",
            data=final_cleaned_df.to_csv(index=False),
            file_name="cleaned_output.csv",
            mime="text/csv",
        )
