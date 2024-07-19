import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def split_dataset_into_train_test(df, random_state):
    '''
        Split dataset into train and test set using stratification.
        
        Args:
            df: dataset
        
        Return:
            X_train, X_test, y_train, y_test: train test sets
    '''
    
    X = np.array(df.drop(columns=['Inactive']))
    y = np.array(df['Inactive'])
    
    # Split the dataset into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True, stratify=y)
    
    return X_train, X_test, y_train, y_test
    
def optimizer(X_train, y_train, random_state):
    '''
        Perform hyper-parameter tinung using RandomizedSearchCV to find best parameters.
        
        Args:
            X_train: train features
            y_train: train target features
        
        Return:
            best parameters
    '''
    # Initialize the Random Forest model
    model = RandomForestClassifier(random_state=random_state)

    # Define the parameter distribution
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'bootstrap': [True, False]
    }

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, scoring='accuracy', random_state=random_state)

    # Fit RandomizedSearchCV to the data
    random_search.fit(X_train, y_train)
    
    return random_search.best_params_

def train_model(best_params, X_train, y_train):
    '''
        Create and train a Random Forest Classifier
        
        Args:
            best_params: the best parameters from the hyper-parameter tuning
            X_train: train features
            y_train: train target feature
        
        Return:
            model: trained model
    '''
    model = RandomForestClassifier(n_estimators=best_params['n_estimators'], \
                                   max_depth=best_params['max_depth'], \
                                   min_samples_split=best_params['min_samples_split'], \
                                   min_samples_leaf=best_params['min_samples_leaf'], \
                                   bootstrap=best_params['bootstrap'], \
                                   random_state=46)

    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    '''
        Evaluate model on test dataset.
        
        Args:
            model: trained model
            X_test: test features
            y_test: test target feature
        
        Return:
            y_pred: predictions of the model
    '''
    
    # Make predictions on the testing set
    y_pred = model.predict(X_test)
    
    return y_pred

def print_metrics(y_test, y_pred):
    '''
        Print evaluation metrics
        
        Args:
            y_test: test target feature
            y_pred: predictions of the model
    '''
    print('Evaluation metrics:')
    print('\tAccuracy: {}'.format(round(accuracy_score(y_test, y_pred)*100, 2))) 
    print('\tPrecision: {}'.format(round(precision_score(y_test, y_pred)*100, 2))) 
    print('\tRecall: {}'.format(round(recall_score(y_test, y_pred)*100, 2)))
    print('\tF1-Score: {}'.format(round(f1_score(y_test, y_pred)*100, 2)))
    
def plot_confusion_matrix(y_test, y_pred, results_path, filename):
    '''
        Plot confusion matriox of the model after training and testing
        
        Args:
            y_test: test target feature
            y_pred: predictions of the model
            results_path: the complete path to save confusion matrix
            filename: name of the image file
    '''
    cf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 7))

    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    # Set labels and title
    plt.xlabel('Actual Class')
    plt.ylabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.savefig(results_path + filename)
    plt.show()

if __name__ == "__main__":
    
    # Fix path.
    PATH = '/home/nsintoris/Churn_Prediction/data/final_dataset.csv'
    RANDOM_STATE = 46

    # Load csv file. 
    data_df = pd.read_csv(PATH)
    print('Final dataset: \n{}'.format(data_df))
    
    # Drop CustomerID column.
    data_df = data_df.drop(columns=['CustomerID'])
    print('Final dataset without CustomerID feature: \n{}'.format(data_df))
    
    # Split the dataset into training and testing sets with stratification
    X_train, X_test, y_train, y_test = split_dataset_into_train_test(data_df, RANDOM_STATE)
    
    # Hyper-parameter tuning on random forest classifier.
    best_params = optimizer(X_train, y_train, RANDOM_STATE)
    print('Final parameters after random Search CV: \n{}'.format(best_params))
    
    # Train model.
    model = train_model(best_params, X_train, y_train)
    
    # Evaluate model.
    y_pred = evaluate_model(model, X_test, y_test)
    
    print_metrics(y_test, y_pred)
    
    plot_confusion_matrix(y_test, y_pred, '/home/nsintoris/Churn_Prediction/results/', 'confusion_matrix.png')
    
    print('Finished')