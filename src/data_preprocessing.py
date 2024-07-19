import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def encode_categorical_features(df):
    '''
        Encode categorical features using One-Hot and Label encoding.
        
        Args:
            df: original dataframe
        
        Return:
            df_encoded: dataframe after categorical encoding
    '''
    # One-Hot Encoding on Geography and Gender features.
    df_encoded = pd.get_dummies(data=df, columns=['Geography', 'Gender'])

    # Label Encoding on Age_Band feature.
    label_encoder = LabelEncoder()
    df_encoded['Age_Band'] = label_encoder.fit_transform(df_encoded['Age_Band'])
    
    # Reorder columns.
    df_encoded = df_encoded[['CustomerID', 'Geography_Athens', 'Geography_Rest_GR', 'Geography_Thessaloniki', \
                             'Gender_Female','Gender_Male','Age_Band', 'TenureYears', 'EstimatedIncome',\
                             'BalanceEuros','NoProducts', 'CreditCardholder', 'CustomerWithLoan','Digital_TRX_ratio', 'Inactive']]
    
    return df_encoded

def handle_imbalanced_data(df):
    '''
        Handle imbalanced data using SMOTE technique.
        
        Args:
            df: encoded dataframe
        
        Return:
            resampled_df: dataframe after applying SMOTE technique
    '''
    # Define features (X) and target (y).
    X = df.drop(columns=['Inactive'])
    y = df['Inactive']
    
    # Apply SMOTE technique to balance the dataset.
    smote_obj = SMOTE(sampling_strategy='minority', random_state=42)
    X_resampled, y_resampled = smote_obj.fit_resample(X, y)
    
    # Combine the features and target into a single DataFrame
    resampled_df = pd.concat([X_resampled, y_resampled], axis=1)
    
    return resampled_df

def plot_target_class_distribution(df, results_path, filename):
    '''
        Create a barplot to see target feature distribution.
        
        Args:
            df: data dataframe
            results_path: path where you store results
            filename: the name of the file
        
        Return:
            plot the barplot
    '''
    # Create the bar plot
    plt.figure(figsize=(10, 7))
    barplot = sns.countplot(x='Inactive', data=df)

    # Annotate the counts above each bar
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.0f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Set labels and title
    plt.xlabel('Inactive')
    plt.ylabel('Count')
    plt.title('Count of Active vs. Inactive Customers')
    plt.savefig(results_path + filename)
    plt.show()
    
def save_to_csv(df, path, filename):
    '''
        Save final dataset as a csv.
        
        Args:
            df: final dataframe
            path: path where you store data
            filename: the name of the file
    '''
    # Save DataFrame to CSV
    df.to_csv(path_or_buf=path+filename, sep=',', header=True, index=False)
    print('Dataset saved to {} as csv!'.format(path))

if __name__ == "__main__":
    
     # Fix path.
    PATH = '/home/nsintoris/Churn_Prediction/data/Churn_data.csv'
    
    # Load csv file. 
    data_df = pd.read_csv(PATH)
    print('Original dataset: \n{}'.format(data_df))
    
    # Encode categorical features.
    data_df = encode_categorical_features(data_df)
    print('Dataset after categorical encoding: \n{}'.format(data_df))
    
    # Plot target feature distribution before SMOTE.
    plot_target_class_distribution(data_df, '/home/nsintoris/Churn_Prediction/results/', 'target_feature_distributions_plot.png')
    
    # Apply SMOTE.
    resampled_df = handle_imbalanced_data(data_df)
    print('Dataset after SMOTE technique: \n{}'.format(resampled_df))
    
    # Plot target feature distribution after SMOTE.
    plot_target_class_distribution(resampled_df,'/home/nsintoris/Churn_Prediction/results/', 'target_feature_distributions_plot_after.png')
    
    # Save final dataset after preprocessing.
    save_to_csv(df=resampled_df, path='/home/nsintoris/Churn_Prediction/data/', filename='final_dataset.csv')
    
    print('Finished!')

    