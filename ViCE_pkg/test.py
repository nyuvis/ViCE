import pandas as pd
import numpy as np

import data
import model

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
# How to deal with the preprocessing? Generate and then delete after?



path = "diabetes.csv"
model_path = 'finalized_model.sav'

d = data.Data(path)
m = model.Model().load_model(model_path)
# m.train_model(d, type='svm')

d2 = data.Data(example = "diabetes")




# --- Setting random seed -- 
# np.random.seed(150)

# --- Parameters --- 
# data_path = "static/data/delinquency/delinquency.csv"
# preproc_path = "static/data/delinquency/delinquency_preproc.csv"
# data_path = "static/data/heart/heart.csv"
# preproc_path = "static/data/heart/heart_preproc.csv"
# no_bins = 10
# model_path = "TBD"   # Manual? 


# # --- Advanced Parameters
# density_fineness = 1000
# categorical_cols = []  # Categorical columns can be customized
# monotonicity_arr = []


# df = pd.read_csv(data_path)
# feature_names = np.array(df.columns)[:-1]
# all_data = np.array(df.values)

# # -- Split data and target values --
# data = all_data[:,:-1]
# target = all_data[:,-1]

# no_samples, no_features = data.shape

# svm_model = SVM_model(data,target)
# svm_model.train_model(0.001)
# svm_model.test_model()

# index = 7

# test_sample = data[index]
# 
# bins_centred, X_pos_array, init_vals, col_ranges = divide_data_bins(data,no_bins)  # Note: Does not account for categorical features
# 
# single_bin_result = bin_single_sample(test_sample, col_ranges)

# aggr_data = prep_for_D3_aggregation(preproc_path, data, feature_names, [0,1,2,3,4,10], bins_centred, X_pos_array, False)



# density_fineness = 1000
# all_den, all_median, all_mean = all_kernel_densities(data,feature_names,density_fineness) # Pre-load density distributions
# cols_lst = [3,9,11,12]

# anchs = False
# print(ids_with_combination(preproc_path, cols_lst, anchs))
# sample_no = 1
# locked = [1,2,3]

# monotonicity_arr = mono_finder(svm_model, data, col_ranges)
# print("MONOTONICITY ARRAY:")
# print(monotonicity_arr)
# change_vector, change_row, anchors, percent = instance_explanation(svm_model, data, data[sample_no], sample_no, X_pos_array, bins_centred, 
#                                     no_bins, monotonicity_arr, col_ranges)
# for s in range(no_samples):
#     change_vector, change_row, anchors, percent = instance_explanation(svm_model, data, data[s], s, X_pos_array, bins_centred, 
#                                                     no_bins, monotonicity_arr, col_ranges)
#     print(change_vector)
# instance_explanation(svm_model, data, row, sample, X_pos_array,
#             bins_centred, no_bins, monotonicity_arr, col_ranges)
# create_summary_file(data, target, svm_model, bins_centred, X_pos_array, init_vals, no_bins, monotonicity_arr, preproc_path)
# res = prepare_for_D3(data[sample_no], bins_centred, change_row, change_vector, anchors, percent, feature_names, False, monotonicity_arr)



if __name__ == '__main__':

    from preprocessing import create_summary_file

    
    # --- Setting random seed -- 
    np.random.seed(150)

    # --- Parameters --- 
    # data_path = "static/data/delinquency/delinquency.csv"
    # preproc_path = "static/data/delinquency/delinquency_preproc.csv"
    data_path = "static/data/heart/heart.csv"
    preproc_path = "static/data/heart/heart_preproc.csv"
    no_bins = 10
    model_path = "TBD"   # Manual? 


    # --- Advanced Parameters
    density_fineness = 1000
    categorical_cols = []  # Categorical columns can be customized
    # monotonicity_arr = []


    df = pd.read_csv(data_path)
    feature_names = np.array(df.columns)[:-1]
    all_data = np.array(df.values)

    # -- Split data and target values --
    data = all_data[:,:-1]
    target = all_data[:,-1]

    no_samples, no_features = data.shape

    svm_model = SVM_model(data,target)
    svm_model.train_model(0.001)
    svm_model.test_model()

    index = 7

    # test_sample = data[index]

    bins_centred, X_pos_array, init_vals, col_ranges = divide_data_bins(data,no_bins)  # Note: Does not account for categorical features
    
    # single_bin_result = bin_single_sample(test_sample, col_ranges)

    # aggr_data = prep_for_D3_aggregation(preproc_path, data, feature_names, [0,1,2,3,4,10], bins_centred, X_pos_array, False)



    # density_fineness = 1000
    # all_den, all_median, all_mean = all_kernel_densities(data,feature_names,density_fineness) # Pre-load density distributions
    cols_lst = [3,9,11,12]

    anchs = False
    # print(ids_with_combination(preproc_path, cols_lst, anchs))
    sample_no = 1
    # locked = [1,2,3]

    monotonicity_arr = mono_finder(svm_model, data, col_ranges)
    # print("MONOTONICITY ARRAY:")
    # print(monotonicity_arr)
    # change_vector, change_row, anchors, percent = instance_explanation(svm_model, data, data[sample_no], sample_no, X_pos_array, bins_centred, 
    #                                     no_bins, monotonicity_arr, col_ranges)
    # for s in range(no_samples):
    #     change_vector, change_row, anchors, percent = instance_explanation(svm_model, data, data[s], s, X_pos_array, bins_centred, 
    #                                                     no_bins, monotonicity_arr, col_ranges)
    #     print(change_vector)
# instance_explanation(svm_model, data, row, sample, X_pos_array,
#             bins_centred, no_bins, monotonicity_arr, col_ranges)
    # create_summary_file(data, target, svm_model, bins_centred, X_pos_array, init_vals, no_bins, monotonicity_arr, preproc_path)
    # res = prepare_for_D3(data[sample_no], bins_centred, change_row, change_vector, anchors, percent, feature_names, False, monotonicity_arr)






    
 


    