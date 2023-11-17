from sklearn.impute import SimpleImputer ## Handling missing value
from sklearn.preprocessing import StandardScaler #Handling feature Scalling
from sklearn.preprocessing import OrdinalEncoder #Ordenal encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

# Data transformation config
@dataclass
class DataTransformationconfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

# Data ingestion class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformation()
    
    def get_data_transformation_object(self):

        try:
            logging.info('Data transformation initiated')
            # Define which columns should be treated with which transformer
            logging.info('pipeline initiated')
            #Numerical pipeline
            num_pipeline = Pipeline(
                                steps=[
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())
                                ]
                                )


            # Create the preprocessor
            preprocessor = ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols)
            ])
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:

            logging.info("Error in data transformation")
            raise CustomException(e,sys)

    def initiate_data_transformation (self,train_path,path,test_path):
        try:
            #reading train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data is completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')
            
            logging.info('obtaining preprocessor object')

            preprocessing_obj=self.get_data_transformation_object()

            target_column_name='heating_load_(HL)','cooling_load_(CL)'
            drop_columns=[target_column_name]

            # feature into dependent and independent feature

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            # apply the trainsformation

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprosessingobject on training and testing datasets")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                
            )
        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation")

            raise CustomException(e,sys)