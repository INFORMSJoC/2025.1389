import os
import time
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np

from apps import *
from library.utils_c import *
from library.ocrt_c import OCRT
from library.RandomForest import RandomForestOCRT
import itertools
from pathlib import Path

base_folder = os.getcwd()

def weighted_sum_squared_error(y_test, y_pred, w):
    """
    Calculates sum_i sum_j w_j * (y_pred_j - y_test_ij)^2
    """    
    squared_diff = (y_test - y_pred) ** 2    
    weighted_row_sums = squared_diff @ w
    return np.sum(weighted_row_sums)

if __name__ == '__main__':
    ocrt_min_samples_split = 10
    ocrt_min_samples_leaf = 5
    number_of_folds = 5
    verbose = False
    bforce_list = [True]
    use_hashmaps_list = [True]    
    use_initial_solution_list = [False]
    ocrt_depth_list = [5,7] 
    class_target_size_list = [5]
    class_size_list = [500]
    seed_list = [i for i in range(3)]
    dataset_list = ['synthetic_manifold'] # synthetic_manifold
    evaluation_method_list = ['mse'] # mse, mad, poisson
    prediction_method_leaf_list = ['optimal'] # optimal
    prediction_method_list = ['mean','optimal'] # mean, medoid, optimal
    n_estimator = 20 # Number of Random Forest regressors
    
    perf_df = pd.DataFrame()
    
    extra_params = list(itertools.product(class_size_list, class_target_size_list,seed_list))
    result = [(dataset, ocrt_depth, c_size, c_target,s)
    for dataset, ocrt_depth in itertools.product(dataset_list, ocrt_depth_list)
    for c_size, c_target,s in (extra_params if (dataset == 'synthetic_manifold' or 
          dataset == 'synthetic_illustrative' or dataset == 'hts') else [(None, None, None)])]
    
    for dataset, ocrt_depth, class_size, class_target_size,s in result:
        if dataset == 'synthetic_manifold':
            optimization_problem = formulate_and_solve_lp_transhipment_e2e
            problem = 'synthetic_manifold'
            target_cols = [f'y{id + 1}' for id in range(class_target_size)]

            feature_cols = ['X1','X2','X3','X4','X5','X6']
            file_name = f'noisefree_df_size_{class_size}_targets_{class_target_size}_features_6_seed_{s}.csv'

            full_df = pd.read_csv(f'{base_folder}/data/noisefree_df_size_{class_size}_targets_{class_target_size}_features_6_seed_{s}.csv',dtype='float')

            features_df = full_df[feature_cols]
            targets_df = full_df[target_cols]
            weights = np.ones(class_target_size)

            targetsum = targets_df.mul(weights).sum(axis=1)
            print('Weights',weights)
            datasetname = f'noisefree_df_size_{class_size}_targets_{class_target_size}_features_6_seed_{s}'
        else:
            raise ValueError('Dataset Error: Dataset is not properly defined')

        num_rows = features_df.shape[0]
        num_targets = targets_df.shape[1]

        kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=10)
        for cv_fold, (tr_idx, te_idx) in enumerate(kf.split(features_df)):
            print('************************')
            print(f'Fold: {cv_fold}')     
            
            # One-hot encoding for categorical features
            if dataset in ['hts','hts_global']:
                categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns
                features_df = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True, dtype=int)

            X_train, y_train, ysum_train = features_df.iloc[tr_idx], targets_df.iloc[tr_idx], targetsum.iloc[tr_idx]
            X_test, y_test, ysum_test = features_df.iloc[te_idx], targets_df.iloc[te_idx], targetsum.iloc[te_idx]
            
            regressor = DecisionTreeRegressor(random_state=20, min_samples_leaf=ocrt_min_samples_leaf,
                                            min_samples_split=ocrt_min_samples_split, max_depth=ocrt_depth)
            
            cap = 1000
            quantities, optcost= {},{}
            for index, row in targets_df.iterrows():
                ytemp = row.tolist()
                quantities[index],optcost[index] = formulate_and_solve_lp_transhipment(ytemp,capacity=cap)
                
            quantities_df = pd.DataFrame.from_dict(quantities, orient='index')
            quantities_train= quantities_df.iloc[tr_idx]
            quantities_test = quantities_df.iloc[te_idx]
            
            
            # Sklearn regression for each output then compute mse using ysum
            start = time.time()
            regressor.fit(X_train, y_train)
            end = time.time()
            y_pred_sklearn = regressor.predict(X_train)
            dt_mse_sum_train = weighted_sum_squared_error(y_train, y_pred_sklearn, weights)
            print(f'DT MSE sum train: {dt_mse_sum_train}')
            
            y_pred_sklearn = regressor.predict(X_test)
            dt_mse_sum = weighted_sum_squared_error(y_test, y_pred_sklearn,weights)
            print(f'DT MSE sum test: {dt_mse_sum}')
            
            # Sklearn regression for each output and compute mse using y vector
            y_pred_sklearn = regressor.predict(X_train)
            start = time.time()
            end = time.time()
            dt_mse_train = mean_squared_error(y_train, y_pred_sklearn)
            print(f'DT MSE train: {dt_mse_train}')
            
            y_pred_sklearn = regressor.predict(X_test)
            dt_mse = mean_squared_error(y_test, y_pred_sklearn)
            print(f'DT MSE test: {dt_mse}')
            
            y_pred_df = pd.DataFrame(y_pred_sklearn)
            y_pred_df['leaf_id'] = regressor.apply(X_test)
            
            y_pred_sklearn_df = y_pred_df
            
            quantitiessk, optcostsk= {},{}
            cost = [0 for i in range(len(y_pred_sklearn))]
            regretsk = [0 for i in range(len(y_pred_sklearn))]
            for i in range(len(y_pred_sklearn)):
                leaf = y_pred_df.at[i,'leaf_id']
                if leaf in quantitiessk:
                    temp = 0
                    for t in range(class_target_size):
                        cost[i] += y_test.iloc[i].tolist()[t]*quantitiessk[leaf][t]
                        temp += y_test.iloc[i].tolist()[t]*quantities_test.iloc[i].tolist()[t]
                    regretsk[i] += abs(temp-cost[i])**2
                else:
                    quantitiessk[leaf],optcostsk[leaf] = formulate_and_solve_lp_transhipment(y_pred_sklearn[i],capacity=cap)
                    temp = 0
                    for t in range(class_target_size):
                        cost[i] += y_test.iloc[i].tolist()[t]*quantitiessk[leaf][t]
                        temp += y_test.iloc[i].tolist()[t]*quantities_test.iloc[i].tolist()[t]
                    regretsk[i] += abs(temp-cost[i])**2
            
            
            avg_sklearn_cost = sum(cost)/len(cost)
            avg_sklearn_regret2 = sum(regretsk)/len(regretsk)
            avg_sklearn_regret = (sum([abs(aa) for aa in regretsk])**.5)/len(regretsk)
            print('Cost Sklearn',avg_sklearn_cost)
            print('Regret Sklearn',avg_sklearn_regret)

            
            dt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred_sklearn, X_test, dataset, 'DT',
                                                                        regressor.get_depth(), target_cols)

            perf_df = pd.concat([perf_df, pd.DataFrame({'data': [datasetname], 'fold': [cv_fold], 'depth': [ocrt_depth],
                                                        'weight_seed':[s],
                                                        'min_samples_leaf': [ocrt_min_samples_leaf],
                                                        'min_samples_split': [ocrt_min_samples_split],
                                                        'prediction_method': ['sklearn'],
                                                        'prediction_method_leaf': ['sklearn'],
                                                        'evaluation_method': ['sklearn'],
                                                        'mse_sum': [dt_mse_sum], 'mse': [dt_mse], 
                                                        'mse gap sum':[0], 'mse gap':[0],
                                                        'optimization_cost':[avg_sklearn_cost], 'optimization_cost_gap':[0],
                                                        'optimization_regret':[avg_sklearn_regret], 'optimization_regret_gap':[0],
                                                        'optimization_regret2':[avg_sklearn_regret2], 'optimization_regret2_gap':[0],
                                                        'training_duration': [end - start]})], ignore_index=True)

            extra_params2 = list(itertools.product(use_hashmaps_list, use_initial_solution_list, evaluation_method_list,prediction_method_list,prediction_method_leaf_list))
            prod = [(bforce, use_hashmaps, use_initial_solution, evaluation_method,prediction_method,prediction_method_leaf)
            for bforce in bforce_list
            for use_hashmaps, use_initial_solution, evaluation_method,prediction_method,prediction_method_leaf
            in (extra_params2 if bforce else list(itertools.product([False], use_initial_solution_list, ['mse'], ['sing-depthMIP'], ['sing-depthMIP'])))]

            for bforce, use_hashmaps, use_initial_solution, evaluation_method,prediction_method,prediction_method_leaf in prod:

                print("==============")
                print(f'Dataset: {datasetname}', f'ocrt_depth:{ocrt_depth}')
                print(f'Seed: {s}')
                print(f'Split Prediction: {prediction_method}')
                print(f'Leaf Prediction: {prediction_method_leaf}')
                
                
                nof_infeasibilities_method = lambda y, x: calculate_number_of_infeasibilities(y, x, dataset,
                                                                        'OCRT', ocrt_depth, target_cols, verbose)
                lagrangian_multiplier = 0
                
                # w, quantities and capacity arrive from the tree (see the ocrt
                # constructor below), so use those parameters rather than closing
                # over the outer weights/cap. They are forwarded as keyword-only
                # extras, which keeps the positional signature of
                # split_criteria_with_methods identical to the other experiments.
                split_criteria = lambda y, x, w, quantities, capacity, nof_infeasibilities_method, initial_solution: \
                    split_criteria_with_methods(y, x, nof_infeasibilities_method,
                            initial_solution, lagrangian_multiplier, prediction_method, evaluation_method, optimization_problem, verbose, bforce,
                            w=w, quantities=quantities, capacity=capacity)
                
                leaf_prediction_method = lambda y, x, w, quantities, capacity, nof_infeasibilities_method, initial_solution: \
                    split_criteria_with_methods(y, x, nof_infeasibilities_method,
                            initial_solution, lagrangian_multiplier, prediction_method_leaf, evaluation_method, optimization_problem, verbose, bforce,
                            w=w, quantities=quantities, capacity=capacity)
                
                # NOTE: ocrt_c must be extended to accept and forward quantities and
                # capacity alongside w before this runs (see the note at the top).
                tree = OCRT(max_depth=ocrt_depth, min_samples_leaf=ocrt_min_samples_leaf, min_samples_split=ocrt_min_samples_split,
                            w = weights, pass_w = True, quantities = quantities_train, capacity = cap,
                            split_criteria=split_criteria, leaf_prediction_method=leaf_prediction_method,
                            nof_infeasibilities_method=nof_infeasibilities_method, verbose=verbose, use_hashmaps = use_hashmaps, use_initial_solution = use_initial_solution,
                            bforce=bforce,problem=problem)
                
                tree.fit(X_train, y_train)
                y_pred_train, quantity_train = tree.predict(X_train)
                
                y_pred, quantity_pred = tree.predict(X_test)
                ocrt_mse_sum = weighted_sum_squared_error(y_test, y_pred,weights)
                ocrt_mse = mean_squared_error(y_train, y_pred_train)
                ocrt_mse = mean_squared_error(y_test, y_pred)
                
                y_pred_df = pd.DataFrame(y_pred)
                y_pred_df['leaf_id'] = tree.apply(X_test)[0]
                
                quantity_pred_df = pd.DataFrame(quantity_pred)
                quantity_pred_df['leaf_id'] = tree.apply(X_test)[0]
                
                # renamed: this used to shadow the per-fold `quantities` dict built
                # from targets_df above, which quantities_train/_test come from
                leaf_quantities, optcostsk = {},{}
                cost = [0 for i in range(len(y_pred))]
                regret = [0 for i in range(len(y_pred))]
                for i in range(len(y_pred)):
                    leaf = y_pred_df.at[i,'leaf_id']
                    if leaf not in leaf_quantities:
                        leaf_quantities[leaf] = quantity_pred_df.iloc[i,:-1]
                    temp = 0
                    for t in range(class_target_size):
                        cost[i] += y_test.iloc[i].tolist()[t]*leaf_quantities[leaf][t]
                        temp += y_test.iloc[i].tolist()[t]*quantities_test.iloc[i].tolist()[t]
                    regret[i] += abs(temp-cost[i])**2
                
                
                avg_ocrt_cost = sum(cost)/len(cost)
                avg_ocrt_regret2 = sum(regret)/len(regret)
                avg_ocrt_regret = (sum([abs(aa) for aa in regret])**.5)/len(regret)
                print('Cost OCRT',avg_ocrt_cost)
                print('Regret Sklearn',avg_sklearn_regret)

                print('Regret OCRT',avg_ocrt_regret)
                
                print('Cost',avg_sklearn_cost,avg_ocrt_cost,100*(avg_sklearn_cost-avg_ocrt_cost)/avg_sklearn_cost)
                print('Regret',avg_sklearn_regret,avg_ocrt_regret,100*(avg_sklearn_regret-avg_ocrt_regret)/avg_sklearn_regret)
                
                ocrt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred, X_test, dataset, 'OCRT', ocrt_depth, target_cols)
                
                
                perf_df = pd.concat([perf_df, pd.DataFrame({'data': [datasetname], 'fold': [cv_fold], 'depth': [ocrt_depth],
                                                            'min_samples_leaf': [ocrt_min_samples_leaf],
                                                            'min_samples_split': [ocrt_min_samples_split],
                                                            'prediction_method': [prediction_method],
                                                            'prediction_method_leaf': [prediction_method_leaf],
                                                            'evaluation_method': [evaluation_method],
                                                            # 'mse': [ocrt_mse], 'Noise free mse': [ocrt_mse2], 
                                                            'mse_sum': [ocrt_mse_sum], 'mse': [ocrt_mse], 
                                                            'mse gap sum':[(ocrt_mse_sum-dt_mse_sum)/dt_mse_sum], 'mse gap':[(ocrt_mse-dt_mse)/dt_mse],
                                                            'optimization_cost':[avg_ocrt_cost], 'optimization_cost_gap':[(avg_sklearn_cost-avg_ocrt_cost)/avg_sklearn_cost],
                                                            'optimization_regret':[avg_ocrt_regret], 'optimization_regret_gap':[(avg_sklearn_regret-avg_ocrt_regret)/avg_sklearn_regret],
                                                            'optimization_regret2':[avg_ocrt_regret2], 'optimization_regret2_gap':[(avg_sklearn_regret2-avg_ocrt_regret2)/avg_sklearn_regret2],
                                                            'nof_infeasibilities': [ocrt_nof_infeasibilities],
                                                            'training_duration': [tree.training_duration],
                                                            'use_brute_force': [bforce],
                                                            'use_hashmaps': [use_hashmaps],
                                                            'use_initial_solution': [use_initial_solution]})], ignore_index=True)


                # perf_df.to_csv(f'data/results/{datasetname}_e2e.csv', index=False)
                print(f'Training Time of OCRT: {tree.training_duration}')
                
              