import os
import sys
import pickle

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model_with_gridsearch(X_train, X_test, y_train, y_test, models, param_grids, cv=5):
    report = {}

    for name, model in models.items():
        # Get the parameter grid for the current model
        param_grid = param_grids.get(name, {})

        # Perform GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Best model from GridSearchCV
        best_model = grid_search.best_estimator_

        # Fit the best model on the full training data
        best_model.fit(X_train, y_train)

        # Predict Testing data using the best model
        y_test_pred = best_model.predict(X_test)

        # Calculate test accuracy
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Store test accuracy in the report
        report[name] = test_accuracy

        # Print results
        print(f"{name} Testing Accuracy: {test_accuracy:.4f}")
        print(f"{name} Best Parameters: {grid_search.best_params_}")
        print("-" * 40)

    return report

def evaluate_model(X_train, y_train, X_test, y_test, models, param_grids, cv=5):
    try:
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import GridSearchCV

        return evaluate_model_with_gridsearch(X_train, X_test, y_train, y_test, models, param_grids, cv)

    except Exception as e:
        logging.info('Exception occurred during model training')
        raise CustomException(e, sys)
