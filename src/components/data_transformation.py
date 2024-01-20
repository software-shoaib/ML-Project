import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer #used to create a pipeline
from sklearn.impute import SimpleImputer #used for the missing values
from sklearn.pipeline import Pipeline #used to create a pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler #used for the encoding and scaling
from src.exception import CustomException
from src.logger import logging 
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preproccessor_obj_file_path:str=os.path.join('artifacts','preprocessor_obj.pkl')    

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_tranformer_object(self):

        '''
        This function is responsible for data transformation 

        '''
        try:
            numerical_columns = ["writing score","reading score"]
            categorical_columns =[
                "gender",
                "race_ethinicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")), #handling missing vales through median via imputer and median is used for outliers
                ("scaler",StandardScaler()) #doing standard scaling
                ]
            )

            cat_pipeline = Pipeline(
                 steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")), #handling missing values through most frequent via imputer
                ("one_hot_encodde", OneHotEncoder()), #doing one hot encoding for less values it i used
                ("scaler", StandardScaler()) #doing standard scaling
                 ]

            )
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            #for combining numerical and categorical columns we use column transformer

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
            
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completed")
            logging.info("obtaining preproccessing object")

            preproccessing_obj=self.get_data_tranformer_object()

            target_column_name="math score"

            numerical_columns = ["writing score","reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preproccessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr=preproccessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preproccessing_obj.transform(input_feature_test_df)
            

            train_arr =np.c_[
                input_feature_train_arr,np.array (target_feature_train_df)
            ]

            test_arr =np.c_[
                input_feature_test_arr,np.array (target_feature_test_df)
            ]

            logging.info("Saving preproccessing object")

            save_object(
                file_path=self.data_transformation_config.preproccessor_obj_file_path,
                obj=preproccessing_obj

            )
            

            return(
                train_arr,
                test_arr,
                self.transformation_config.preproccessor_obj_file_path
            )            
        except Exception as e:
            raise CustomException(e,sys)