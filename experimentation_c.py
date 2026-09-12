import os
import time
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np

from apps import *
from library.utils import *
from library.ocdt_c import OCDT
from library.RandomForest import RandomForestOCDT
from library.Preprocessing import *
import itertools

base_folder = os.getcwd()

if __name__ == '__main__':
    ocdt_min_samples_split = 10
    ocdt_min_samples_leaf = 5
    number_of_folds = 5
    verbose = False
    bforce_list = [True,False] # True for EOCRT - False for MOCRT
    use_hashmaps_list = [True]    
    use_initial_solution_list = [False]
    ocdt_depth_list = [5,7] 
    class_target_size_list = [5,9]
    class_size_list = [500,1000,2000]
    feature_size_list = [6]
    seed_list = [i for i in range(3)]
    dataset_list = ['synthetic_manifold'] # synthetic_manifold, class, hts
    evaluation_method_list = ['mse'] # mse, mad, poisson
    prediction_method_leaf_list = ['optimal'] # medoid, optimal
    prediction_method_list = ['mean','optimal'] # mean, medoid, optimal
    n_estimator = 20 # Number of Random Forest regressors

    
    perf_df = pd.DataFrame()
    
    extra_params = list(itertools.product(class_size_list, class_target_size_list,seed_list,feature_size_list))
    result = [(dataset, ocdt_depth, c_size, c_target,s,featuresize)
    for dataset, ocdt_depth in itertools.product(dataset_list, ocdt_depth_list)
    for c_size, c_target,s,featuresize in (extra_params if (dataset == 'synthetic_manifold' or dataset == 'synthetic_manifold_nonconvex' or 
          dataset == 'hts' or dataset == 'hts_global') else [(None, None, None)])]
    
    for dataset, ocdt_depth, class_size, class_target_size,s,featuresize in result:
        # holds the feasible-set projection of the targets; only the manifold
        # datasets have one defined, so it stays None for the others
        preprop_targets_df = None

        if dataset == 'class':
            optimization_problem = formulate_and_solve_lp_class_data
            problem = 'class'
            target_cols = [f'Course{id + 1}' for id in range(class_target_size)]
            lagrangian_multiplier = 1500

            feature_cols = ['EnrolledElectiveBefore', 'GradeAvgPrevElec',
                            'Grade', 'Major', 'Class', 'GradePerm']

            full_df = pd.read_csv(f'{base_folder}/data/class_df_size_{class_size}_targets_{class_target_size}.csv')
            features_df = full_df[feature_cols]
            targets_df = full_df[target_cols]
            targets_df = targets_df.apply(lambda col: col / 100)
            features_df = features_df.apply(lambda col: col / 100)
            datasetname = f'class_df_size_{class_size}_targets_{class_target_size}'
        elif dataset == 'synthetic_manifold':
            optimization_problem = formulate_and_solve_lp_synthetic_manifold_data
            problem = 'synthetic_manifold'
            target_cols = [f'y{id + 1}' for id in range(class_target_size)]

            feature_cols = [f'X{id + 1}' for id in range(featuresize)] 

            full_df = pd.read_csv(f'{base_folder}/data/synthetic_manifold_df_size_{class_size}_targets_{class_target_size}_features_{featuresize}_seed_{s}.csv',dtype='float')

            features_df = full_df[feature_cols]
            targets_df = full_df[target_cols]

            # project each target vector onto the feasible set
            preprop_df = project_dataframe_targets(full_df, class_size, featuresize, class_target_size)
            preprop_targets_df = preprop_df[target_cols]

            datasetname = f'synthetic_manifold_df_size_{class_size}_targets_{class_target_size}_features_{featuresize}_seed_{s}'
        elif dataset == 'hts':
            optimization_problem = formulate_and_solve_lp_hts_data
            problem = 'hts'
            target_cols = [f'y{id + 1}' for id in range(13)]

            feature_cols = ['X1','X2','X3','X4','X5','X6']
            
            
            full_df = pd.read_csv(f'{base_folder}/data/hts_df_size_{class_size}_seed_{s}_v2.csv',dtype='float')
            features_df = full_df[feature_cols]
            numeric_cols = features_df.select_dtypes(include=['number']).columns
            targets_df = full_df[target_cols]
            datasetname = f'hts_df_size_{class_size}_seed_{s}_v2'
        elif dataset == 'forecasting':
            optimization_problem = formulate_and_solve_lp_forecasting_data
            problem = 'forecasting'
            full_df = pd.read_csv(f'{base_folder}/data/forecasting.csv')
            feature_cols = [f'Feature {id + 1}' for id in range(30)]
            target_cols = [f'Target {id + 1}' for id in range(6)]
            lagrangian_multiplier = 1000
            features_df = full_df[feature_cols]
            targets_df = full_df[target_cols]
            datasetname = dataset
        elif dataset == 'scores':
            optimization_problem = formulate_and_solve_lp_scores_data
            problem = 'scores'
            feature_cols = ['gender', 'race/ethnicity', 'parental level of education',
                            'lunch', 'test preparation course']
            target_cols = ['math score', 'reading score', 'writing score']
            lagrangian_multiplier = 0
            full_df = pd.read_csv(f'{base_folder}/data/constrained_exams.csv')
            
            features_df = full_df[feature_cols]
            targets_df = full_df[target_cols]
            features_df = pd.get_dummies(features_df, columns=features_df.columns, drop_first=True, dtype=int)
            
            targets_df = targets_df.apply(lambda col: col / 100)
            datasetname = dataset

        else:
            optimization_problem = formulate_and_solve_lp_cars_data
            problem = 'cars'
            full_df = pd.read_csv(f'{base_folder}/data/insurance_evaluation_data.csv').drop(columns=['INDEX']).dropna()[:500]

            target_cols = ['TARGET_FLAG', 'TARGET_AMT']
            targets_df = full_df[target_cols]
            lagrangian_multiplier = 20000000
            feature_cols = ['KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ', 'INCOME', 'HOME_VAL', 'TRAVTIME',
                            'BLUEBOOK', 'TIF', 'OLDCLAIM', 'CLM_FREQ', 'MVR_PTS', 'CAR_AGE']
            features_df = full_df[feature_cols]
            currency_cols = features_df.select_dtypes('object').columns
            features_df.loc[:, currency_cols] = features_df[currency_cols].replace(r'[$,]', '', regex=True).astype(float)
            datasetname = dataset

        num_rows = features_df.shape[0]
        num_targets = targets_df.shape[1]

        kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=10)
        for cv_fold, (tr_idx, te_idx) in enumerate(kf.split(features_df)):
            print('************************')
            print(f'Fold: {cv_fold}')            
            

            X_train, y_train = features_df.iloc[tr_idx], targets_df.iloc[tr_idx]
            X_test, y_test = features_df.iloc[te_idx], targets_df.iloc[te_idx]

            # the projection touches targets only, so the features are shared;
            # the test targets stay raw so every model is scored identically
            has_projection = preprop_targets_df is not None
            preprop_y_train = preprop_targets_df.iloc[tr_idx] if has_projection else None
            
            regressor = DecisionTreeRegressor(random_state=20, min_samples_leaf=ocdt_min_samples_leaf,
                                            min_samples_split=ocdt_min_samples_split, max_depth=ocdt_depth)
            start = time.time()
            regressor.fit(X_train, y_train)
            end = time.time()

            y_pred_sklearn = regressor.predict(X_train)
            dt_mse_train = mean_squared_error(y_train, y_pred_sklearn)

            print(f'DT MSE_train: {dt_mse_train}')
            y_pred_sklearn = regressor.predict(X_test)
            dt_mse = mean_squared_error(y_test, y_pred_sklearn)
            print(f'DT MSE: {dt_mse}')
            
            dt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred_sklearn, X_test, dataset, 'DT',
                                                                        regressor.get_depth(), target_cols)

            perf_df = pd.concat([perf_df, pd.DataFrame({'data': [datasetname], 'fold': [cv_fold], 'depth': [ocdt_depth],
                                                        'min_samples_leaf': [ocdt_min_samples_leaf],
                                                        'min_samples_split': [ocdt_min_samples_split],
                                                        'prediction_method': ['sklearn'],
                                                        'prediction_method_leaf': ['sklearn'],
                                                        'evaluation_method': ['sklearn'],
                                                        'mse': [dt_mse], 'mse gap':[0],
                                                        'nof_infeasibilities': [dt_nof_infeasibilities],
                                                        'training_duration': [end - start]})], ignore_index=True)

            # ---- same DT, fitted on the projected training targets ----------
            if has_projection:
                preprop_regressor = DecisionTreeRegressor(random_state=20, min_samples_leaf=ocdt_min_samples_leaf,
                                                min_samples_split=ocdt_min_samples_split, max_depth=ocdt_depth)
                start = time.time()
                preprop_regressor.fit(X_train, preprop_y_train)
                end = time.time()

                preprop_y_pred_sklearn = preprop_regressor.predict(X_train)
                preprop_dt_mse_train = mean_squared_error(preprop_y_train, preprop_y_pred_sklearn)
                print(f'Preprocessed DT MSE_train: {preprop_dt_mse_train}')

                preprop_y_pred_sklearn = preprop_regressor.predict(X_test)
                preprop_dt_mse = mean_squared_error(y_test, preprop_y_pred_sklearn)
                print(f'Preprocessed DT MSE: {preprop_dt_mse}')

                preprop_dt_nof_infeasibilities = calculate_number_of_infeasibilities(preprop_y_pred_sklearn, X_test, dataset, 'DT',
                                                                            preprop_regressor.get_depth(), target_cols)

                perf_df = pd.concat([perf_df, pd.DataFrame({'data': [datasetname], 'fold': [cv_fold], 'depth': [ocdt_depth],
                                                            'min_samples_leaf': [ocdt_min_samples_leaf],
                                                            'min_samples_split': [ocdt_min_samples_split],
                                                            'prediction_method': ['preprop_sklearn'],
                                                            'prediction_method_leaf': ['preprop_sklearn'],
                                                            'evaluation_method': ['preprop_sklearn'],
                                                            'mse': [preprop_dt_mse], 'mse gap':[(preprop_dt_mse-dt_mse)/dt_mse],
                                                            'nof_infeasibilities': [preprop_dt_nof_infeasibilities],
                                                            'training_duration': [end - start]})], ignore_index=True)

            extra_params2 = list(itertools.product(use_hashmaps_list, use_initial_solution_list, evaluation_method_list,prediction_method_list,prediction_method_leaf_list))
            prod = [(bforce, use_hashmaps, use_initial_solution, evaluation_method,prediction_method,prediction_method_leaf)
            for bforce in bforce_list
            for use_hashmaps, use_initial_solution, evaluation_method,prediction_method,prediction_method_leaf
            in (extra_params2 if bforce else list(itertools.product([False], use_initial_solution_list, ['mse'], ['sing-depthMIP'], ['sing-depthMIP'])))]

            for bforce, use_hashmaps, use_initial_solution, evaluation_method,prediction_method,prediction_method_leaf in prod:

                print("==============")
                print(f'Dataset: {datasetname}', f'OCDT_depth:{ocdt_depth}')
                print(f'Brute Force: {bforce}')
                print(f'Initial Solution: {use_initial_solution}')
                print(f'Evaluation: {evaluation_method}')
                print(f'Split Prediction: {prediction_method}')
                print(f'Leaf Prediction: {prediction_method_leaf}')
                
                
                nof_infeasibilities_method = lambda y, x: calculate_number_of_infeasibilities(y, x, dataset,
                                                                        'OCDT', ocdt_depth, target_cols, verbose)
                lagrangian_multiplier = 0
                # no w -> unweighted criterion (original behaviour, unchanged call)
                split_criteria = lambda y, x, nof_infeasibilities_method, initial_solution: split_criteria_with_methods(y, x, nof_infeasibilities_method,
                        initial_solution, lagrangian_multiplier, prediction_method, evaluation_method, optimization_problem, verbose, bforce)
                leaf_prediction_method = lambda y, x, nof_infeasibilities_method, initial_solution: split_criteria_with_methods(y, x, nof_infeasibilities_method,
                        initial_solution, lagrangian_multiplier, prediction_method_leaf, evaluation_method, optimization_problem, verbose, bforce)
                
                # w is unused here: the criteria functions take the 4-argument
                # signature (y, x, nof_infeasibilities_method, initial_solution),
                # so pass_w=False tells the merged OCDT not to forward w.
                tree = OCDT(max_depth=ocdt_depth, min_samples_leaf=ocdt_min_samples_leaf, min_samples_split=ocdt_min_samples_split,
                            w=None, pass_w=False,
                            split_criteria=split_criteria, leaf_prediction_method=leaf_prediction_method,
                            nof_infeasibilities_method=nof_infeasibilities_method, verbose=verbose, use_hashmaps = use_hashmaps, use_initial_solution = use_initial_solution,
                            bforce=bforce,problem=problem)
                
                tree.fit(X_train, y_train)
                y_pred = tree.predict(X_train)
                ocdt_mse_train = mean_squared_error(y_train, y_pred)

                print(f'OCDT MSE Train: {ocdt_mse_train}')
                y_pred = tree.predict(X_test)
                y_pred_df = pd.DataFrame(y_pred)
                y_pred_df['leaf_id'] = tree.apply(X_test)[0]
                y_pred_df = y_pred_df.drop_duplicates()
                ocdt_mse = mean_squared_error(y_test, y_pred)
                print(f'OCDT MSE: {ocdt_mse}')
                
                ocdt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred, X_test, dataset, 'OCDT', ocdt_depth, target_cols)
                
                
                perf_df = pd.concat([perf_df, pd.DataFrame({'data': [datasetname], 'fold': [cv_fold], 'depth': [ocdt_depth],
                                                            'min_samples_leaf': [ocdt_min_samples_leaf],
                                                            'min_samples_split': [ocdt_min_samples_split],
                                                            'prediction_method': [prediction_method],
                                                            'prediction_method_leaf': [prediction_method_leaf],
                                                            'evaluation_method': [evaluation_method],
                                                            'mse': [ocdt_mse], 'mse gap':[(ocdt_mse-dt_mse)/dt_mse],
                                                            'nof_infeasibilities': [ocdt_nof_infeasibilities],
                                                            'training_duration': [tree.training_duration],
                                                            'use_brute_force': [bforce],
                                                            'use_hashmaps': [use_hashmaps],
                                                            'use_initial_solution': [use_initial_solution]})], ignore_index=True)
                
                
                ocdt_params = {
                    "max_depth": ocdt_depth,
                    "w": None,
                    "pass_w": False,
                    "min_samples_leaf": ocdt_min_samples_leaf,
                    "min_samples_split": ocdt_min_samples_split,
                    "split_criteria": split_criteria,
                    "leaf_prediction_method": leaf_prediction_method,
                    "nof_infeasibilities_method": nof_infeasibilities_method,
                    "verbose": verbose,
                    "use_hashmaps": use_hashmaps,
                    "use_initial_solution": use_initial_solution,
                    "bforce": bforce,
                    "problem": problem
                }
                _, m = X_train.shape
                forest = RandomForestOCDT(n_estimators=n_estimator, random_state=42, max_features=int(m*0.8), **ocdt_params)
                forest.fit(X_train, y_train)
                y_pred = forest.predict(X_test,num_targets)
                RFocdt_mse = mean_squared_error(y_test, y_pred)
                print(f'RF MSE: {RFocdt_mse}')
                RFocdt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred, X_test, dataset, 'RF_OCDT', ocdt_depth, target_cols)
                
                perf_df = pd.concat([perf_df, pd.DataFrame({'data': [datasetname], 'fold': [cv_fold], 'depth': [ocdt_depth],
                                                            'min_samples_leaf': [ocdt_min_samples_leaf],
                                                            'min_samples_split': [ocdt_min_samples_split],
                                                            'prediction_method': [prediction_method],
                                                            'prediction_method_leaf': [prediction_method_leaf],
                                                            'evaluation_method': [evaluation_method],
                                                            'mse': [RFocdt_mse], 'mse gap':[(RFocdt_mse-dt_mse)/dt_mse],
                                                            'nof_infeasibilities': [RFocdt_nof_infeasibilities],
                                                            'training_duration': [forest.training_duration],
                                                            'use_brute_force': [bforce],
                                                            'use_hashmaps': [use_hashmaps],
                                                            'use_initial_solution': [use_initial_solution],
                                                            'RF_n_estimators': [n_estimator]})], ignore_index=True)
                
                
                # ---- same forest, fitted on the projected training targets --
                if has_projection:
                    preprop_forest = RandomForestOCDT(n_estimators=n_estimator, random_state=42, max_features=int(m*0.8), **ocdt_params)
                    preprop_forest.fit(X_train, preprop_y_train)
                    y_pred = preprop_forest.predict(X_test,num_targets)
                    preprop_RFocdt_mse = mean_squared_error(y_test, y_pred)
                    print(f'RF MSE preprocessed: {preprop_RFocdt_mse}')
                    preprop_RFocdt_nof_infeasibilities = calculate_number_of_infeasibilities(y_pred, X_test, dataset, 'RF_OCDT', ocdt_depth, target_cols)

                    perf_df = pd.concat([perf_df, pd.DataFrame({'data': [datasetname], 'fold': [cv_fold], 'depth': [ocdt_depth],
                                                                'min_samples_leaf': [ocdt_min_samples_leaf],
                                                                'min_samples_split': [ocdt_min_samples_split],
                                                                'prediction_method': ['preprop_' + prediction_method],
                                                                'prediction_method_leaf': ['preprop_' + prediction_method_leaf],
                                                                'evaluation_method': [evaluation_method],
                                                                'mse': [preprop_RFocdt_mse], 'mse gap':[(preprop_RFocdt_mse-dt_mse)/dt_mse],
                                                                'nof_infeasibilities': [preprop_RFocdt_nof_infeasibilities],
                                                                'training_duration': [preprop_forest.training_duration],
                                                                'use_brute_force': [bforce],
                                                                'use_hashmaps': [use_hashmaps],
                                                                'use_initial_solution': [use_initial_solution],
                                                                'RF_n_estimators': [n_estimator]})], ignore_index=True)

                # perf_df.to_csv(f'data/results/perf_df_{datasetname}_df_new_experimentation.csv', index=False)

                print(f'Training Time of OCDT: {tree.training_duration}')
                print(f'Training Time of RF: {forest.training_duration}')
                if has_projection:
                    print(f'Training Time of RF preprocessed: {preprop_forest.training_duration}')
                