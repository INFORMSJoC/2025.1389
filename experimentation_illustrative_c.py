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
import itertools

base_folder = os.getcwd()

if __name__ == '__main__':
    ocrt_min_samples_split = 10
    ocrt_min_samples_leaf = 5
    number_of_folds = 5
    verbose = False
    bforce_list = [True]
    use_hashmaps_list = [True]    
    use_initial_solution_list = [False]
    ocrt_depth_list = [5] 
    class_target_size_list = [5]
    class_size_list = [500]
    seed_list = [i for i in range(3)]
    dataset_list = ['synthetic_illustrative'] # synthetic_illustrative
    evaluation_method_list = ['mse'] # mse, mad, poisson
    prediction_method_leaf_list = ['optimal'] # medoid, optimal
    prediction_method_list = ['mean','optimal'] # mean, medoid, optimal
    
    perf_df = pd.DataFrame()
    
    extra_params = list(itertools.product(class_size_list, class_target_size_list,seed_list,seed_list))
    result = [(dataset, ocrt_depth, c_size, c_target,s,ss)
    for dataset, ocrt_depth in itertools.product(dataset_list, ocrt_depth_list)
    for c_size, c_target,s,ss in (extra_params if (dataset == 'synthetic_manifold' or 
          dataset == 'synthetic_illustrative' or dataset == 'hts') else [(None, None, None,None)])]
    
    for dataset, ocrt_depth, class_size, class_target_size,s,ss in result:
        if dataset == 'synthetic_illustrative':
            # optimization_problem = formulate_and_solve_lp_synthetic_manifold_data
            optimization_problem = formulate_and_solve_lp_synthetic_illustrative_data
            problem = 'synthetic_illustrative'
            target_cols = [f'y{id + 1}' for id in range(class_target_size)]

            feature_cols = ['X1','X2','X3','X4','X5','X6']

            full_df = pd.read_csv(f'{base_folder}/data/synthetic_manifold_df_size_{class_size}_targets_{class_target_size}_features_6_seed_{s}.csv',dtype='float')
            features_df = full_df[feature_cols]
            targets_df = full_df[target_cols]
            np.random.seed(ss)
            weights = np.random.rand(class_target_size)
            targetsum = targets_df.mul(weights).sum(axis=1)
            datasetname = f'synthetic_illustrative_df_size_{class_size}_targets_{class_target_size}_seed_{s}'
            
        else:
            raise ValueError('Dataset Error: Dataset is not properly defined')

        num_rows = features_df.shape[0]
        num_targets = targets_df.shape[1]

        kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=10)
        for cv_fold, (tr_idx, te_idx) in enumerate(kf.split(features_df)):
            print('************************')
            print(f'Fold: {cv_fold}')            
            
            X_train, y_train, ysum_train = features_df.iloc[tr_idx], targets_df.iloc[tr_idx], targetsum.iloc[tr_idx]
            X_test, y_test, ysum_test = features_df.iloc[te_idx], targets_df.iloc[te_idx], targetsum.iloc[te_idx]
            regressor = DecisionTreeRegressor(random_state=20, min_samples_leaf=ocrt_min_samples_leaf,
                                            min_samples_split=ocrt_min_samples_split, max_depth=ocrt_depth)
            
            # Sklearn regression for each output then compute mse using ysum
            start = time.time()
            regressor.fit(X_train, y_train)
            end = time.time()
            y_pred_sklearn = regressor.predict(X_train)
            ysum_pred_sklearn = (y_pred_sklearn * weights).sum(axis=1)
            dt_mse_sum_train = mean_squared_error(ysum_train, ysum_pred_sklearn)
            print(f'DT MSE_train: {dt_mse_sum_train}')
            
            y_pred_sklearn = regressor.predict(X_test)
            # ysum_pred_sklearn = y_pred_sklearn.sum(axis=1)
            ysum_pred_sklearn = (y_pred_sklearn * weights).sum(axis=1)
            dt_mse_sum = mean_squared_error(ysum_test, ysum_pred_sklearn)
            print(f'DT MSE: {dt_mse_sum}')
            
            # Sklearn regression for each output and compute mse using y vector
            y_pred_sklearn = regressor.predict(X_train)
            start = time.time()
            end = time.time()
            dt_mse_train = mean_squared_error(y_train, y_pred_sklearn)
            print(f'DT MSE_train: {dt_mse_train}')
            
            y_pred_sklearn = regressor.predict(X_test)
            dt_mse = mean_squared_error(y_test, y_pred_sklearn)
            print(f'DT MSE: {dt_mse}')
            
            
            perf_df = pd.concat([perf_df, pd.DataFrame({'data': [datasetname], 'fold': [cv_fold], 'depth': [ocrt_depth],
                                                        'weight_seed':[ss],
                                                        'min_samples_leaf': [ocrt_min_samples_leaf],
                                                        'min_samples_split': [ocrt_min_samples_split],
                                                        'prediction_method': ['sklearn'],
                                                        'prediction_method_leaf': ['sklearn'],
                                                        'evaluation_method': ['sklearn'],
                                                        'mse_sum': [dt_mse_sum], 'mse': [dt_mse], 
                                                        'mse gap sum':[0], 'mse gap':[0],
                                                        'training_duration': [end - start]})], ignore_index=True)
            
            extra_params2 = list(itertools.product(use_hashmaps_list, use_initial_solution_list, evaluation_method_list,prediction_method_list,prediction_method_leaf_list))
            prod = [(bforce, use_hashmaps, use_initial_solution, evaluation_method,prediction_method,prediction_method_leaf)
            for bforce in bforce_list
            for use_hashmaps, use_initial_solution, evaluation_method,prediction_method,prediction_method_leaf
            in (extra_params2 if bforce else list(itertools.product([False], use_initial_solution_list, ['mse'], ['sing-depthMIP'], ['sing-depthMIP'])))]

            for bforce, use_hashmaps, use_initial_solution, evaluation_method,prediction_method,prediction_method_leaf in prod:

                print("==============")
                print(f'Dataset: {datasetname}', f'ocrt_depth:{ocrt_depth}')
                print(f'Seed: {ss}')
                print(f'Split Prediction: {prediction_method}')
                print(f'Leaf Prediction: {prediction_method_leaf}')
                
                
                nof_infeasibilities_method = lambda y, x: calculate_number_of_infeasibilities(y, x, dataset,
                                                                        'ocrt', ocrt_depth, target_cols, verbose)
                lagrangian_multiplier = 0
                # w arrives from the tree (ocrt is constructed with w=weights below),
                # so use it instead of closing over the outer `weights` variable.
                split_criteria = lambda y, x, w, nof_infeasibilities_method, initial_solution: split_criteria_with_methods(y, x, nof_infeasibilities_method,
                        initial_solution, lagrangian_multiplier, prediction_method, evaluation_method, optimization_problem, verbose, bforce, w=w)
                leaf_prediction_method = lambda y, x, w, nof_infeasibilities_method, initial_solution: split_criteria_with_methods(y, x, nof_infeasibilities_method,
                        initial_solution, lagrangian_multiplier, prediction_method_leaf, evaluation_method, optimization_problem, verbose, bforce, w=w)
                
                tree = OCRT(max_depth=ocrt_depth, min_samples_leaf=ocrt_min_samples_leaf, min_samples_split=ocrt_min_samples_split,
                            w = weights, pass_w=True,
                            split_criteria=split_criteria, leaf_prediction_method=leaf_prediction_method,
                            nof_infeasibilities_method=nof_infeasibilities_method, verbose=verbose, use_hashmaps = use_hashmaps, use_initial_solution = use_initial_solution,
                            bforce=bforce,problem=problem)
                
                tree.fit(X_train, y_train)
                y_pred = tree.predict(X_train)
                # ysum_pred = y_pred.sum(axis=1)
                ysum_pred = (y_pred * weights).sum(axis=1)
                
                ocrt_mse_sum_train = mean_squared_error(ysum_train, ysum_pred)
                print(f'OCRT MSE Train: {ocrt_mse_sum_train}')
                
                
                y_pred = tree.predict(X_test)
                # ysum_pred = y_pred.sum(axis=1)
                ysum_pred = (y_pred * weights).sum(axis=1)
                ysum_pred_df = pd.DataFrame(ysum_pred)
                ysum_pred_df['leaf_id'] = tree.apply(X_test)[0]
                # ysum_pred_df = ysum_pred_df.drop_duplicates()
                ocrt_mse_sum = mean_squared_error(ysum_test, ysum_pred)
                print(f'ocrt MSE: {ocrt_mse_sum}')
                ocrt_mse = mean_squared_error(y_test, y_pred)
                
                ocrt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred, X_test, dataset, 'OCRT', ocrt_depth, target_cols)
                
                
                perf_df = pd.concat([perf_df, pd.DataFrame({'data': [datasetname], 'fold': [cv_fold], 'depth': [ocrt_depth],
                                                            'weight_seed':[ss],
                                                            'min_samples_leaf': [ocrt_min_samples_leaf],
                                                            'min_samples_split': [ocrt_min_samples_split],
                                                            'prediction_method': [prediction_method],
                                                            'prediction_method_leaf': [prediction_method_leaf],
                                                            'evaluation_method': [evaluation_method],
                                                            'mse_sum': [ocrt_mse_sum], 'mse': [ocrt_mse], 
                                                            'mse gap sum':[(ocrt_mse_sum-dt_mse_sum)/dt_mse_sum], 
                                                            'mse gap':[(ocrt_mse-dt_mse)/dt_mse],
                                                            # 'mse': [ocrt_mse], 'mse gap':[(ocrt_mse-dt_mse)/dt_mse],
                                                            'training_duration': [tree.training_duration],
                                                            'use_brute_force': [bforce],
                                                            'use_hashmaps': [use_hashmaps],
                                                            'use_initial_solution': [use_initial_solution]})], ignore_index=True)
                
                # perf_df.to_csv(f'data/results/perf_df_{datasetname}_df_experimentation1.csv', index=False)
                # print(f'Training Time of ocrt: {tree.training_duration}')
                
                
                