#Basic Import
import numpy as np
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models={
            "Logistic Regression":LogisticRegression(),
            "Decision Tree Classifier":DecisionTreeClassifier(),
            "Random Forest Classifier": RandomForestClassifier(),
            "Support Vector Classifier": SVC(),
            "Ada Boost Classifier":AdaBoostClassifier(),
            "Gradient Boost Classifier":GradientBoostingClassifier()
        }
            # Define parameter grids for each model
            param_grids = {
             "Logistic Regression": {
             "C": [0.1, 1.0, 10.0],
             "penalty": ["l1", "l2"] },

            "Decision Tree Classifier": {
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]},
             
            "Random Forest Classifier": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]},

            "Support Vector Classifier": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"]},

            "Ada Boost Classifier": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0]},

            "Gradient Boost Classifier": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
             "max_depth": [3, 5, 7]
        }

}

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models,param_grids,cv=5)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , acuuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)