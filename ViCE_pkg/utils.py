import numpy as np
import pandas as pd
from sklearn.neighbors.kde import KernelDensity
from sklearn import preprocessing
from operator import itemgetter
import copy

# class dataset():

# 	def __init__ (self, name, lock):
# 		self.name = name
# 		self.lock = lock


def mono_finder(model, data, ranges):
	"""
	Assumes linearity. 
	Potentially can be used for weighted monotonicity
	"""
	np.random.seed(5)
	
	no_ft = data.shape[1]

	monotonicity_arr = np.zeros(no_ft)

	sample_no = np.random.randint(0,data.shape[0])
	test_sample = data[sample_no]

	for ft in range(no_ft):
		sample_low = copy.deepcopy(test_sample) 
		sample_high = copy.deepcopy(test_sample) 
		
		min_val, max_val  = find_feature_range(ranges[ft])

		sample_low[ft] = min_val
		sample_high[ft] = max_val

		model_low = model.run_model(sample_low)
		model_high = model.run_model(sample_high)

		if model_high > model_low:
			monotonicity_arr[ft] = 1

		else:
			monotonicity_arr[ft] = -1

	return monotonicity_arr

def find_feature_range(feat_range):
	min_val = feat_range[0][0]
	max_val = 0  
	for i in range(len(feat_range)):  # Finding the max range value
		if (i == len(feat_range)-1):
			max_val = feat_range[i][1]

		elif (feat_range[i+1] == '-1'):
			max_val = feat_range[i][1]
			break
	return (min_val,max_val)

def model_overview(pre_proc_file):
	pre_data = pd.read_csv(pre_proc_file).values

	total_count = pre_data.shape[0]
	
	changes_count = 0
	key_count = 0

	tp_count = 0
	fp_count = 0

	tn_count = 0
	fn_count = 0

	for sample in pre_data:

		if sample[2]== "TP":
			tp_count += 1

		elif sample[2]== "FP":
			fp_count += 1
 
		elif sample[2] == "TN":
			tn_count += 1

		elif sample[2] == "FN":
			fn_count += 1


		if sample[3] > 0:
			key_count += 1

		if sample[4] > 0:
			changes_count += 1


	# print("-- Model Summary --")

	# print("Total # of samples:", total_count)
	# print()
	# print("True Positive:",tp_count)
	# print("False Positive:",fp_count)
	# print("True Negative:",tn_count)
	# print("False Negative:",fn_count)
	# print()
	# print("Key Features:",key_count)
	# print("Changes",changes_count)

def separate_bins_feature(feat_column, no_bins):

	# -- All other cases --
	feat_column = feat_column.flatten()
	two_std = 2*np.std(feat_column)
	avg_val = np.mean(feat_column)

	# -- Finding the Range --
	if (avg_val - two_std < 0):
		min_val = 0
	else:
		min_val = round((avg_val - two_std),0)
	max_val = round((avg_val + two_std),0)

	# -- Creating the Bins --
	single_bin = (max_val - min_val) // no_bins
	if (single_bin == 0):
		single_bin = 1
	
	centre = min_val + (single_bin // 2)
	floor = min_val
	ceil = min_val + single_bin

	ranges = []
	str_ranges = []
	bins = np.zeros(no_bins)
	new_col = np.zeros(feat_column.shape[0])
	new_col_vals = np.zeros(feat_column.shape[0])

	for i in range(no_bins):
		range_str = ""
		if (centre <= max_val):
			for val_i in range(feat_column.shape[0]):
					if (i == 0):
						range_str = "x < " + str(ceil)
						if (feat_column[val_i] < ceil):
							new_col[val_i] = i
							new_col_vals[val_i] = centre

					elif (i == no_bins-1) or ((centre + single_bin) > max_val):
						range_str = str(floor) + " < x"
						if (feat_column[val_i] >= floor):
							new_col[val_i] = i
							new_col_vals[val_i] = centre

					else:
						range_str = str(floor) +" < x < " + str(ceil)
						if ((ceil > feat_column[val_i]) and (feat_column[val_i] >= floor)):
							new_col[val_i] = i
							new_col_vals[val_i] = centre
			bins[i] = centre
			str_ranges.append(range_str)
			ranges.append((floor,ceil))
		
		else:
			bins[i] = -1
			str_ranges.append("-1")
			ranges.append("-1")

		

		floor += single_bin
		ceil += single_bin
		centre += single_bin

	return bins, new_col, new_col_vals, ranges

def divide_data_bins(data,no_bins):
    no_feat = data.shape[1]
    bins_centred = []
    X_pos_array = []
    in_vals = []
    col_ranges = []
    
    for i in range(no_feat):
        # Handles special case
        bins, new_col, val, col_range = separate_bins_feature(data[:,i].flatten(),no_bins)
        
        in_vals.append(val)
        bins_centred.append(bins)
        X_pos_array.append(new_col)
        col_ranges.append(col_range)
        
    # Convert to numpy array
    in_vals = np.array(in_vals).transpose()
    bins_centred = np.array(bins_centred)
    X_pos_array = (np.array(X_pos_array)).transpose()
    col_ranges = np.array(col_ranges)

    return bins_centred, X_pos_array, in_vals, col_ranges

def bin_single_sample(sample, col_ranges):
	# -- Extract Basic Parameters -- 
	no_features = len(sample)
	no_bins = len(col_ranges[0])
	pos_array = np.ones(no_features)*-1


	for col_i in range(no_features):
		value = sample[col_i]
		ranges = col_ranges[col_i]

		for bin_no in range(no_bins):
			floor = ranges[bin_no][0]
			ceil = ranges[bin_no][1]
			# -- Dealing with edge cases -- 


			if bin_no == no_bins-1:
				pos_array[col_i] = bin_no
				break

			elif ranges[bin_no + 1] == '-1':
				# if value >= floor:
				pos_array[col_i] = bin_no
				break

			elif bin_no == 0:
				if value <= floor:
					pos_array[col_i] = bin_no
					break

			else:
				if (value < ceil) and (value >= floor):
					pos_array[col_i] = bin_no
					break


	return pos_array

def sort_by_val(main, density):
    ordered_main = []
    ordered_density = []

    ordered_main = sorted(main, key=itemgetter('scl_val'), reverse=True) 
    keySort = sorted(range(len(main)), key = lambda k: main[k]["scl_val"], reverse=True)

    for key in keySort:
        ordered_density.append(density[key])

    return ordered_main, ordered_density

def sort_by_imp(main, density, ft_imps):
	ordered_main = []
	ordered_density = []
	abs_ft_imp = np.abs(ft_imps)

	keySort = np.flip(np.argsort(abs_ft_imp))
	for key in keySort:
		ordered_density.append(density[key])
		ordered_main.append(main[key])

	return ordered_main, ordered_density






