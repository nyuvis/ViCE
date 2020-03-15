# --- Instance Level Explanation --- 

import pandas as pd
import numpy as np
from model import *
from utils import *
from global_explanations import *
from d3_functions import *



class Vice:

    def __init__(self, data, model, mode = "basic", no_bins = 10):

        self.data = data
        self.model = model
        self.mode = mode
        self.no_bins = no_bins


        # --- Advanced Parameters
        density_fineness = 100


        # bins_centred, X_pos_array, init_vals, col_ranges = divide_data_bins(data,no_bins)  



    def generate_explanation(self, index):
        pass








    def __evaluate_data_set(self, data):
        no_features = data.shape[1]
        avg_list = []
        std_list = []
        for i in range(no_features):
            current_col = data[:,i].flatten()
            std_list.append(np.std(current_col))
            avg_list.append(np.mean(current_col))
              
        return avg_list, std_list

    def __perturb_special(self, min_val,max_val,avg,std,no_val):  # Dealing with categorical features
        new_col = np.random.normal(avg, std, no_val)
        # Note: these functions have poor time complexity
        np.place(new_col,new_col < min_val, min_val)
        np.place(new_col,new_col > max_val, max_val)
        new_col = new_col.round(0)
        return new_col
        
    def __find_anchors(self, model, data_set, sample, no_val, special_cols = []):
        # Special Cols account for the categorical columns

        # --- Hardcoded Parameters --- 
        iterations = 4   # Iterations allowed
        

        # --- Manually ---  # Dealing with categoricals. Assigning category range. 
        lowest_category = 0
        highest_category = 7

        
        features = sample.shape[0]
        avg_list, std_list = evaluate_data_set(data_set)

        # Precision Treshold
        treshold = 0.95
        
        # Identify original result from sample
        initial_percentage = model.run_model(sample)
        decision = np.round(initial_percentage,0)

        # Create empty mask 
        mask = np.zeros(features)
        
        # Allows tracking the path
        locked = []


        while (iterations > 0):
            # Retains best result and the corresponding index
            max_ind = (0,0)

            # Assign column that is being tested
            for test_col in range(features):
                new_data = np.empty([features, no_val])

                # Perturb data
                for ind in range(features):
                    if (ind == test_col) or (ind in locked):
                        new_data[ind] = np.array(np.repeat(sample[ind],no_val))
                    else:
                        if (ind in special_cols):

                            new_data[ind] = perturb_special(lowest_category,highest_category,avg_list[ind],std_list[ind],no_val)
                        else:
                            new_data[ind] = np.random.normal(avg_list[ind], std_list[ind], no_val)

                
                new_data = new_data.transpose()


                # Run Model 
                pred = model.run_model_data(new_data)
                acc = (np.mean(pred == decision))
                
                if (acc > max_ind[0]):
                    max_ind = (acc,test_col)
                    

            locked.append(max_ind[1])
                
            for n in locked:
                mask[n] = 1
                
            if (max_ind[0] >= treshold):
                return mask
            iterations -= 1
            
        # print("!!! No anchors found !!!")
        return None

    def __perturb_row_feature(self, model, row, row_idx, feat_idx, current_bins, X_bin_pos, mean_bins, mono_arr, improve, no_bins, col_ranges):
        
        monot_arr = np.copy(mono_arr)                        
        
        c_current_bins = np.copy(current_bins)
        direction = monot_arr[feat_idx]
        current_bin = np.copy(c_current_bins[feat_idx])
        
        if current_bin != no_bins-1:
            next_value = mean_bins[feat_idx][int(current_bin+1)]
        if current_bin < no_bins-2:
            n_next_value = mean_bins[feat_idx][int(current_bin+2)]
        if current_bin != 0:
            prev_value = mean_bins[feat_idx][int(current_bin-1)]
        
        # Set direction for boundary cases
        if direction == -1:
            if current_bin == 0:
                direction = 1
            elif current_bin == no_bins-1 or next_value == -1:
                direction = 0

        # Check if in boundary and return the same row
        if direction == 1:
            if current_bin == no_bins-1 or next_value == -1:
                return (row, c_current_bins)
        elif direction == 0 and current_bin ==  0:
                return (row, c_current_bins)

        # Does not allow for changes into or from last bin (outliers of more than 2 std devs)
        if direction == 1 and current_bin == no_bins-2:
            return (row, c_current_bins)
        elif direction == 1 and n_next_value == -1:
            return (row, c_current_bins)
        if direction == 0 and current_bin == no_bins-1:
            return (row, c_current_bins)
        elif direction == 0 and next_value == -1:
            return (row, c_current_bins)

        
        # Decide direction in special case
        if direction == -1:
            row_up = np.copy(row)
            row_down = np.copy(row)
            row_up[feat_idx] = next_value
            row_down[feat_idx] = prev_value
            percent_up = model.run_model(row_up)
            percent_down = model.run_model(row_down)
            if percent_up >= percent_down:
                if improve:
                    c_current_bins[feat_idx] += 1
                    return (row_up, c_current_bins)
                else:
                    c_current_bins[feat_idx] -= 1
                    return (row_down, c_current_bins)
            elif not improve:
                c_current_bins[feat_idx] -= 1
                return (row_down, c_current_bins)
            else:
                c_current_bins[feat_idx] += 1
                return (row_up, c_current_bins)
            
        else:
            p_row = np.copy(row)
            if direction == 1:
                c_current_bins[feat_idx] += 1
                p_row[feat_idx] = next_value
            elif direction == 0:
                c_current_bins[feat_idx] -= 1
                p_row[feat_idx] = prev_value
            
            return (p_row, c_current_bins)
          

    def __percent_cond (self, improve, percent):
        if improve and percent <= 0.5:
            return True
        elif (not improve) and percent > 0.5:
            return True
        else:
            return False

    def __find_MSC (self, model, data, k_row, row_idx, X_bin_pos, mean_bins, no_bins, monotonicity_arr, col_ranges, keep_top, threshold, locked_fts):

        # --- Hardcoded Parameters --- 
        no_vertical_movement = 5
        no_lateral_movement = 5

        no_features = k_row.shape[0]
        orig_row = np.copy(k_row)
        orig_percent = model.run_model(orig_row)
        orig_moving_fts = np.nonzero(np.array( [1 if not (i in locked_fts) else 0 for i in range(no_features)] ))[0].tolist()

        original_bins = bin_single_sample(orig_row, col_ranges)
        current_bins = bin_single_sample(orig_row, col_ranges)
        
        # --- Decides class to attempt to change into ---
        improve = True
        if orig_percent >= .5:
            improve = False
            
        """
        --- Monotonicity needs to be manually imputed ---
        1: Move up to to improve
        -1: Move down to improve
        0: Needs check

        """

        # mono_finder(model, data, col_ranges)

        if monotonicity_arr == []:
            monotonicity_arr = np.zeros(no_features)
        elif not improve:
            monotonicity_arr *= -1

        top_percents = np.full(keep_top, orig_percent)
        top_rows = np.tile(orig_row, (keep_top,1))
        top_current_bins = np.tile(current_bins, (keep_top,1))
        top_change_vectors = np.tile(np.zeros(no_features), (keep_top,1))
        top_moving_fts = [orig_moving_fts for i in range(keep_top)]

        # Loop while best changed row not above threshold
        while percent_cond(improve, top_percents[0]):

            poss_top_rows = []
            poss_top_percents = []
            poss_top_curr_bins = []

            # Loop over the current top rows
            for j in range(keep_top):

                new_rows = []
                new_percents = []
                new_curr_bins = []
                top_moving_fts[j] = orig_moving_fts.copy()

                # Once lateral threshold reached, only move features already moved
                if np.count_nonzero(top_change_vectors[j]) == no_lateral_movement:
                    top_moving_fts[j] = top_change_vectors[j].nonzero()[0].tolist()

                # Once vertical threshold reached, stop moving that feature
                to_remove = []
                for idx in top_moving_fts[j]:
                    if abs(top_change_vectors[j][idx]) == no_vertical_movement:
                        to_remove.append(idx)
                top_moving_fts[j] = [e for e in top_moving_fts[j] if e not in to_remove]
                try:
                    top_moving_fts[j].extend([top_moving_fts[j][-1] for e in range(keep_top-len(top_moving_fts[j]))]) # Add extra rows in case no. of moving fts < keep_top
                except:
                    print(row_idx, k_row)

                # Avoids moving locked features
                for i in top_moving_fts[j]:
                    # t_row, t_current_bins = perturb_row_feature2(model, top_rows[j], row_idx, i, top_current_bins[j], X_bin_pos, mean_bins, monotonicity_arr, improve, no_bins, col_ranges)
                    # print(monotonicity_arr)
                    t_row, t_current_bins = perturb_row_feature(model, top_rows[j], row_idx, i, top_current_bins[j], X_bin_pos, mean_bins, monotonicity_arr, improve, no_bins, col_ranges)

                    new_rows.append(t_row)
                    new_percents.append(model.run_model(t_row))
                    new_curr_bins.append(t_current_bins)

                new_rows = np.array(new_rows)
                new_percents = np.array(new_percents)
                new_curr_bins = np.array(new_curr_bins)

                idx_sorted = np.argsort(new_percents)
                if improve:
                    idx_sorted = idx_sorted[::-1]

                idx_sorted = idx_sorted[:keep_top]
                new_rows = new_rows[idx_sorted]
                new_percents = new_percents[idx_sorted]
                new_curr_bins = new_curr_bins[idx_sorted]

                for i in range(keep_top):
                    poss_top_rows.append(new_rows[i])
                    poss_top_percents.append(new_percents[i])
                    poss_top_curr_bins.append(new_curr_bins[i])

            poss_top_rows = np.array(poss_top_rows)
            poss_top_percents = np.array(poss_top_percents)
            poss_top_curr_bins = np.array(poss_top_curr_bins)

            top_idx_sorted = np.argsort(poss_top_percents)
            if improve:
                top_idx_sorted = top_idx_sorted[::-1]

            poss_top_rows = poss_top_rows[top_idx_sorted]
            poss_top_percents = poss_top_percents[top_idx_sorted]
            poss_top_curr_bins = poss_top_curr_bins[top_idx_sorted]

            cnt, j = 1, 1
            curr_idx = 0
            final_idx = [0]
            while (cnt < keep_top):                
                while j < len(poss_top_rows) and np.array_equal(poss_top_rows[curr_idx], poss_top_rows[j]):
                    j += 1
                if j >= len(poss_top_rows):
                    final_idx.extend([final_idx[-1] for i in range(keep_top-cnt)])
                    break
                final_idx.append(j)
                curr_idx = j
                j += 1
                cnt += 1

            final_idx = np.array(final_idx)
            
            if (improve and poss_top_percents[0] > top_percents[0]) or ((not improve) and poss_top_percents[0] < top_percents[0]):
                top_rows = poss_top_rows[final_idx]
                top_percents = poss_top_percents[final_idx]
                top_current_bins = poss_top_curr_bins[final_idx]
                for j in range(keep_top):
                    top_change_vectors[j] = top_current_bins[j] - original_bins
            else:
                break
        
        # print(top_percents)
        if not percent_cond(improve, top_percents[0]):
            return top_change_vectors[:keep_top], top_rows[:keep_top]
        else:
            # print("Decision can't be moved within thresholds:")
            if not threshold:
                return top_change_vectors[:keep_top], top_rows[:keep_top]
            else:
                return np.tile(np.zeros(no_features), (keep_top,1)),np.tile(orig_row, (keep_top,1))

    def __instance_explanation(self, model, data, k_row, row_idx, X_bin_pos, mean_bins, no_bins, mono_arr, col_ranges, keep_top=1, threshold=True, locked_fts=[]):
        
        # -- Toggle this for Print statements -- 
        printing = False


        np.random.seed(11)

        # --- To measure performance ---
        model.model_calls = 0

        initial_percentage = model.run_model(k_row)

        try:
            change_vector, change_row = find_MSC(model, data, k_row, row_idx, X_bin_pos, mean_bins, no_bins, mono_arr, col_ranges, keep_top, threshold, locked_fts)
        except:
            change_vector, change_row = np.tile(np.zeros(k_row.shape[0]), (keep_top,1)),np.tile(k_row, (keep_top,1))

        anchors = find_anchors(model, data, k_row, 100)


        if printing:
            print("Model calls for this explanation:", model.model_calls)
            print(change_vector)
            print(change_row)
            print(anchors)
        

        # Find MSC can return a list of change vectors and a list of change rows
        # They can be kept in memory and then passed to D3 functions as necessary.

        if keep_top>1:
            return change_vector, change_row, anchors, initial_percentage
        
        else:
            return change_vector[0], change_row[0], anchors, initial_percentage

