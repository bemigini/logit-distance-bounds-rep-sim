"""
For calculating inverse variance weighted average of student metrics.
We see each teacher as a separate "measuring instrument" and assume students will have a certain variance for a certain teacher.  

"""



import torch
import pandas as pd 
import numpy as np


# metric_name = 'cca' # for testing

def inverse_var_weighted_avg_and_var(metrics_df: pd.DataFrame, metric_name: str, use_sqrt_val: bool = False):
    """
    The inverse-variance weighted average, \\hat{y}, of observations, y_i, with variances, \\sigma_i^2, is:
    \\hat{y} = \\frac{\\sum_i y_i / \\sigma_i^2}{\\sum_i 1 / \\sigma_i^2}

    The variance of the average is: 
    Var(\\hat{y}) = 1 / ( \\sum_i 1 / \\sigma_i^2 )
    
    :param metrics_df: Dataframe with all recorded metrics for student models
    :param metric_name: Name of metric to do weighted average over
    """
    unique_teacher_seeds = metrics_df['teacher_seed'].unique()

    unique_epochs = metrics_df['epoch'].unique()
    unique_model_types = metrics_df['model_type'].unique()

    iv_avg_var_dict = {}

    for current_model_type in unique_model_types:
        iv_avg_var_dict[current_model_type] = {
            'iv_avg_train_pr_epoch': [],
            'iv_avg_test_pr_epoch': [],
            'iv_avg_var_train_pr_epoch': [],
            'iv_avg_var_test_pr_epoch': []
            }
        
        for current_epoch in unique_epochs:        
            current_metrics_df = metrics_df[(metrics_df['epoch']==current_epoch) & (metrics_df['model_type']==current_model_type)]

            sum_obs_over_var_test = 0
            sum_one_over_var_test = 0
            sum_obs_over_var_train = 0
            sum_one_over_var_train = 0

            for current_teacher in unique_teacher_seeds:
                current_teacher_df = current_metrics_df[(current_metrics_df['teacher_seed']==current_teacher)]
                currrent_metric_test = current_teacher_df[f'{metric_name}_test']
                current_metric_train = current_teacher_df[f'{metric_name}_train']
                if use_sqrt_val:
                    currrent_metric_test = np.sqrt(currrent_metric_test)
                    current_metric_train = np.sqrt(current_metric_train)
                var_for_teacher_test = currrent_metric_test.var()
                var_for_teacher_train = current_metric_train.var()
                num_samples = current_teacher_df.shape[0]

                sum_obs_over_var_test += (currrent_metric_test/var_for_teacher_test).sum()
                sum_one_over_var_test += num_samples/var_for_teacher_test
                sum_obs_over_var_train += (current_metric_train/var_for_teacher_train).sum()
                sum_one_over_var_train += num_samples/var_for_teacher_train
            
            iv_weighted_avg_test = sum_obs_over_var_test/sum_one_over_var_test
            iv_weighted_avg_train = sum_obs_over_var_train/sum_one_over_var_train

            iv_avg_var_test = 1/sum_one_over_var_test
            iv_avg_var_train = 1/sum_one_over_var_train

            iv_avg_var_dict[current_model_type]['iv_avg_test_pr_epoch'].append(iv_weighted_avg_test)
            iv_avg_var_dict[current_model_type]['iv_avg_train_pr_epoch'].append(iv_weighted_avg_train)
            iv_avg_var_dict[current_model_type]['iv_avg_var_test_pr_epoch'].append(iv_avg_var_test)
            iv_avg_var_dict[current_model_type]['iv_avg_var_train_pr_epoch'].append(iv_avg_var_train)


    return iv_avg_var_dict


