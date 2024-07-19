import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_missing_values(df):
    '''
        Find if there are missing values in the dataset
        
        Args:
            df: original dataset
        
        Return:
            a dataframe with the number of missing values on each feature.
    '''
    missing_values_per_column = df.isna().sum()

    print(pd.DataFrame(missing_values_per_column, columns=['No. Missing Values']))
    
def distribution_plots(df):
    '''
        Create a plot with all the histograms for each feature.
        
        Args:
            df: original dataset
    '''
    df.hist(figsize=(20, 15))
    plt.savefig('/home/nsintoris/Churn_Prediction/results/distributions_plot.png')
    plt.show()
    
def plot_correlation_heatmap(df):
    '''
        Create a correlation matrix for the dataset.
        
        Args:
            df: original dataset
    '''
    new_df = df.drop(['CustomerID','Geography','Gender','Age_Band'], axis=1)
    corr_matrix = new_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.savefig('/home/nsintoris/Churn_Prediction/results/corellation_heatmap.png')
    plt.show()
    
def plot_relationship_between_feature_and_target(df, feature_name, result_folder_path, image_name):
    '''
        Create 2 barplots between a feature and the target feature.
        
        Args:
            df: original dataset
            feature_name: the name of the feature
            result_folder_path: the path you want to save the image
            image_name: tha name of the final image
    '''
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.countplot(x=feature_name, data=df, ax=axes[0])
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel('Number of Customers')
    axes[0].set_title('Number of Customers in Each ' +  feature_name)

    # Annotate the counts above each bar
    for p in axes[0].patches:
        axes[0].annotate(format(p.get_height(), '.0f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    sns.countplot(x=feature_name, hue='Inactive', data=df, ax=axes[1])
    axes[1].set_xlabel(feature_name)
    axes[1].set_ylabel('Number of Customers')
    axes[1].set_title('Number of Active/Inactive Customers in Each ' + feature_name)

    # Annotate the counts above each bar for Inactive plot
    for p in axes[1].patches:
        axes[1].annotate(format(p.get_height(), '.0f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(result_folder_path + image_name)
    # Show the plot
    plt.show()
    
def plot_relationship_between_age_and_target(df):
    '''
        Create 2 barplots between Age_Band feature and the target feature.
        
        Args:
            df: original dataset
    '''
    # Define the order of the categories
    age_band_order = ['18-25', '25-35', '35-45', '45-55', '55-65', '65+']

    df_copy = df.copy()

    # Convert Age_Band to a categorical type with the specified order
    df_copy['Age_Band'] = pd.Categorical(df['Age_Band'], categories=age_band_order, ordered=True)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.countplot(x='Age_Band', data=df_copy, ax=axes[0], order=age_band_order)
    axes[0].set_xlabel('Age Band')
    axes[0].set_ylabel('Number of Customers')
    axes[0].set_title('Number of Customers in Each Age Band')

    # Annotate the counts above each bar
    for p in axes[0].patches:
        axes[0].annotate(format(p.get_height(), '.0f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    sns.countplot(x='Age_Band', hue='Inactive', data=df_copy, ax=axes[1], order=age_band_order)
    axes[1].set_xlabel('Age Band')
    axes[1].set_ylabel('Number of Customers')
    axes[1].set_title('Number of Active/Inactive Customers in Each Age Band')

    # Annotate the counts above each bar for Inactive plot
    for p in axes[1].patches:
        axes[1].annotate(format(p.get_height(), '.0f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Adjust layout
    plt.tight_layout()
    plt.savefig('/home/nsintoris/Churn_Prediction/results/age_vs_inactive.png')
    # Show the plot
    plt.show()


if __name__ == "__main__":
    
    # Fix path.
    PATH = '/home/nsintoris/Churn_Prediction/data/Churn_data.csv'
    
    # Load csv file. 
    data_df = pd.read_csv(PATH)
    print('Original dataset: \n{}'.format(data_df))
    
    print('Original dataset stats: \n{}'.format(data_df.describe()))
    
    # Number of unique customers.
    no_unique_costomers = len(pd.unique(data_df['CustomerID']))
    print('Number of unique customers: {}'.format(no_unique_costomers))
    
    find_missing_values(data_df)
    
    distribution_plots(data_df)
    
    plot_correlation_heatmap(data_df)
    
    plot_relationship_between_feature_and_target(data_df, 'Geography', '/home/nsintoris/Churn_Prediction/results/', 'geography_vs_inactive.png')
    
    plot_relationship_between_feature_and_target(data_df, 'Gender', '/home/nsintoris/Churn_Prediction/results/', 'gender_vs_inactive.png')
    
    plot_relationship_between_age_and_target(data_df)
    
    print('Finished')