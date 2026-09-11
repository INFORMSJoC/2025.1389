import inspect

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances

def return_medoid(y):
    return y[np.argmin(euclidean_distances(y).mean(axis=1))]

def return_mean(y):
    return np.asarray(y.mean(axis=0))

def return_median(y):
    return np.asarray(np.median(y,axis=0))

def calculate_mse(y, predictions):
    return ((y - predictions) ** 2).mean()

def calculate_mad(y, predictions):
    return np.mean(np.abs(y - predictions))

def calculate_poisson_deviance(y, predictions):
    return 2 * np.sum(predictions - y - y * np.log(predictions / y))

def calculate_number_of_infeasibilities(y_pred, X_test, dataset, model, ocdt_depth, target_cols, verbose=True):
    if dataset == 'class':
        cumsums = np.array([sum(y_pred[i] > 0.0001) for i in range(len(y_pred))])
        nof_infeasibilities = np.sum(cumsums >= 3)
    elif dataset == 'scores':
        nof_infeasibilities = 0
        for i in range(len(y_pred)):
            if (y_pred[i][1] < .50) and (y_pred[i][2] > 0.0001):
                nof_infeasibilities += 1
            elif (round(y_pred[i][1] + y_pred[i][2], 4) < 1.10) and (y_pred[i][0] > 0.0001):
                nof_infeasibilities += 1
    elif dataset == 'synthetic':
        nof_infeasibilities = 0
        num_targets = len(y_pred[0])
        for i in range(len(y_pred)):
            if abs(sum(y_pred[i]) - num_targets*0.5) > 1e-4:
                nof_infeasibilities += 1
            elif (sum(y_pred[i][j] for j in range(int(num_targets/2))) - 0.4*int(num_targets/2) > 1e-4): 
                nof_infeasibilities += 1
            elif(sum(y_pred[i][j] for j in range(int(num_targets/2)+1,num_targets)) - 0.6*(int(num_targets/2)+1) > 1e-4):
                nof_infeasibilities += 1
    elif dataset == 'synthetic_nonconvex':
        nof_infeasibilities = 0
        num_targets = len(y_pred[0])
        for i in range(len(y_pred)):
            if abs(sum(y_pred[i]) - (num_targets//2)*0.5) > 1e-4:
                nof_infeasibilities += 1
            elif (sum(1 for x in y_pred[i] if x != 0) > num_targets//2): 
                nof_infeasibilities += 1
    elif dataset == 'synthetic_binary':
        nof_infeasibilities = 0
        num_targets = len(y_pred[0])
        nob = num_targets//2
        for i in range(len(y_pred)):
            if abs(sum(y_pred[i]) - nob) > 1e-4:
                nof_infeasibilities += 1
            elif abs(sum(y_pred[i][j] for j in range(int(num_targets/2))) - int(nob/2)) > 1e-4: 
                nof_infeasibilities += 1
            elif abs(sum(y_pred[i][j] for j in range(int(num_targets/2)+1,num_targets)) - (nob - int(nob/2))) > 1e-4:
                nof_infeasibilities += 1
    elif dataset == 'synthetic_manifold':
        nof_infeasibilities = 0
        num_targets = len(y_pred[0])
        for i in range(len(y_pred)):
            if abs(sum(y_pred[i][j] for j in range(num_targets//2)) - 1) > 1e-4:
                nof_infeasibilities += 1
            else:
                t = 1
                for k in range(num_targets//2,num_targets,2):
                    if abs(y_pred[i][k-1] - y_pred[i][k] - (t)/10) > 1e-4:
                        nof_infeasibilities += 1
                        break
                    t+=1 
    elif dataset == 'hts':
        nof_infeasibilities = 0
        num_targets = len(y_pred[0])
        for i in range(len(y_pred)):
            if (sum(1 for x in y_pred[i] if x != 0) > 13-9): 
                nof_infeasibilities += 1
            elif abs(sum(y_pred[i][j] for j in range(num_targets)) - 15) > 1e-4: 
                nof_infeasibilities += 1
            else:
                for j in range(num_targets):
                    # if y_pred[i][j] > 20:
                    #     nof_infeasibilities += 1
                    #     break
                    # elif y_pred[i][j] < 0:
                    #     nof_infeasibilities += 1
                    #     break
                    if y_pred[i][j] < 0:
                        nof_infeasibilities += 1
                        break
                    
    elif dataset == 'hts_global':
        nof_infeasibilities = 0
        num_targets = len(y_pred[0])
        for i in range(len(y_pred)):
            if (sum(1 for x in y_pred[i][:12] if x != 0) > 12-6): 
                nof_infeasibilities += 1
            else:
                if y_pred[i][12] != y_pred[i][0]+y_pred[i][1]+y_pred[i][2]:
                    nof_infeasibilities += 1
                elif y_pred[i][13] != y_pred[i][3]+y_pred[i][4]+y_pred[i][5]:
                    nof_infeasibilities += 1
                elif y_pred[i][14] != y_pred[i][6]+y_pred[i][7]+y_pred[i][8]:
                    nof_infeasibilities += 1
                elif y_pred[i][15] != y_pred[i][9]+y_pred[i][10]+y_pred[i][11]:
                    nof_infeasibilities += 1
                elif y_pred[i][16] != y_pred[i][12]+y_pred[i][13]+y_pred[i][14]+y_pred[i][15]:
                    nof_infeasibilities += 1
                else:
                    for j in range(12):
                        if y_pred[i][j] > 10:
                            nof_infeasibilities += 1
                            break
                        elif y_pred[i][j] < 0:
                            nof_infeasibilities += 1
                            break
    elif dataset == 'forecasting':
        nof_infeasibilities = 0
        nof_infeasibilities += y_pred[y_pred > 100].shape[0]
        y_pred_df = pd.DataFrame(y_pred, index=X_test.index)
        y_pred_df = pd.concat([X_test.iloc[:, -3:], y_pred_df], axis=1)
        y_pred_df.columns = range(y_pred_df.shape[1])
        for i in range(1, y_pred_df.shape[1] - 2):
            nof_infeasibilities += y_pred_df[(y_pred_df.loc[:, [i, i+1]].sum(axis=1) > 70) &
                                             (y_pred_df.loc[:, i+2] > 50)].shape[0]
        for i in range(y_pred_df.shape[1] - 3):
            nof_infeasibilities += y_pred_df[(y_pred_df.loc[:, i: i + 2].sum(axis=1) > 120) &
                                             (y_pred_df.loc[:, i + 3] > 10)].shape[0]
    else:
        nof_infeasibilities = 0
        num_targets = len(y_pred[0])
        for i in range(len(y_pred)):
            if abs(sum(y_pred[i][j] for j in range(num_targets//2)) - 1) > 1e-4:
                nof_infeasibilities += 1
            else:
                t = 1
                for k in range(num_targets//2,num_targets,2):
                    if abs(y_pred[i][k-1] - y_pred[i][k] - (t)/10) > 1e-4:
                        nof_infeasibilities += 1
                        break
                    t+=1 

    if verbose:
        print(f'Number of infeasible predictions for {model} (Depth {ocdt_depth}): {nof_infeasibilities}')

    return nof_infeasibilities

# ---------------------------------------------------------------------------
# How the weight vector w enters the split criterion.
#
# False (default): w is forwarded to the LP formulation only, and the split is
#   evaluated on the full target vector. This matches the original code, where
#   the weighted evaluation line was commented out in favour of
#   calculate_mse(y, predictions). The 'mean' / 'medoid' / 'median' methods are
#   then unaffected by w.
#
# True: the split is evaluated on the scalarised target sum_j w_j * y_j.
# ---------------------------------------------------------------------------
SCALARISED_SPLIT_CRITERION = False

# Parameter names an LP formulation may use for the extra arguments.
WEIGHT_NAMES = ('w', 'weights', 'weight')
QUANTITY_NAMES = ('quantities', 'quantity', 'quantities_df')
CAPACITY_NAMES = ('capacity', 'cap')


def apply_weights(y, predictions, w):
    """
    Map targets and predictions into the space the split is evaluated in.

    Returns them untouched unless SCALARISED_SPLIT_CRITERION is enabled, in
    which case both are collapsed to the weighted sum sum_j w_j * y_j. The
    prediction handed back to the tree is always the full vector; only the
    evaluation is affected.
    """
    if w is None or not SCALARISED_SPLIT_CRITERION:
        return y, predictions

    if callable(w):
        raise TypeError(
            "split_criteria_with_methods received a callable as `w`; pass the "
            "weight vector as the keyword argument w=..."
        )
    w = np.asarray(w, dtype=float)
    try:
        predictions = np.asarray(predictions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "The prediction handed to apply_weights is not a numeric vector "
            f"(got {type(predictions).__name__}). If the LP formulation returns "
            "more than the prediction vector, extend as_prediction_vector so it "
            "is unwrapped there."
        ) from exc
    return np.asarray(y, dtype=float) @ w, predictions @ w


def as_prediction_vector(result, n_targets=None):
    """
    Normalise whatever an LP formulation returns into a flat prediction vector.

    Most formulations return the prediction vector directly; some return it
    bundled with extra information. When a tuple/list comes back, the element
    matching the number of targets is taken, otherwise the first element.
    """
    if isinstance(result, tuple) or (isinstance(result, list)
                                     and len(result) and np.ndim(result[0]) >= 1):
        candidates = list(result)
        if n_targets is not None:
            for item in candidates:
                try:
                    arr = np.asarray(item, dtype=float)
                except (TypeError, ValueError):
                    continue
                if arr.ndim == 1 and arr.shape[0] == n_targets:
                    return arr
        result = candidates[0] if candidates else result

    return np.asarray(result, dtype=float).ravel()


def call_optimization_problem(optimization_problem, y, x, initial_solution, lagrangian_multiplier, verbose,
                              w=None, quantities=None, capacity=None):
    """
    Call an LP formulation, forwarding only the extra arguments it declares.

    The formulations in apps.py differ in how much context they need:

        (y, x, initial_solution, lagrangian_multiplier, verbose)
        (y, x, w, initial_solution, lagrangian_multiplier, verbose)
        (y, x, w, quantities, capacity, initial_solution, lagrangian_multiplier, verbose)

    Rather than assume one of these, the formulation's own parameter names are
    walked in order: a parameter named like a weight/quantity/capacity gets the
    corresponding extra, anything else consumes the next of y, x,
    initial_solution, lagrangian_multiplier, verbose. The raw return value is
    passed back untouched, since some formulations return more than the
    prediction vector.
    """
    base_args = [y, x, initial_solution, lagrangian_multiplier, verbose]
    extras = {}
    if w is not None:
        extras.update({nm: w for nm in WEIGHT_NAMES})
    if quantities is not None:
        extras.update({nm: quantities for nm in QUANTITY_NAMES})
    if capacity is not None:
        extras.update({nm: capacity for nm in CAPACITY_NAMES})

    if not extras:
        return optimization_problem(*base_args)

    try:
        params = [prm for prm in inspect.signature(optimization_problem).parameters.values()
                  if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD)]
        names = [prm.name for prm in params]
    except (TypeError, ValueError):
        names = []

    if not names:
        # introspection unavailable: fall back to the y, x, w, quantities,
        # capacity, ... ordering used throughout this codebase
        args = [y, x]
        if w is not None:
            args.append(w)
        if quantities is not None:
            args.append(quantities)
        if capacity is not None:
            args.append(capacity)
        args += [initial_solution, lagrangian_multiplier, verbose]
        return optimization_problem(*args)

    remaining = iter(base_args)
    args = []
    for name in names:
        if name in extras:
            args.append(extras[name])
        else:
            try:
                args.append(next(remaining))
            except StopIteration:
                break
    return optimization_problem(*args)


def split_criteria_with_methods(y, x, nof_infeasibilities_method, initial_solution, lagrangian_multiplier, prediction_method,
                                evaluation_method, optimization_problem, verbose=False, bforce = True,
                                *, w=None, quantities=None, capacity=None):
    """
    Positional signature is unchanged from the original; w, quantities and
    capacity are keyword-only extras.

    Omit all three for the plain experiments, pass w= for the weighted
    (illustrative) ones, and pass w=, quantities=, capacity= for the
    end-to-end ones. In that last case the return value is the triple
    (predictions, split_evaluation, quantity_hat); otherwise it is the pair
    (predictions, split_evaluation), exactly as before.
    """
    end_to_end = quantities is not None or capacity is not None

    if not bforce:
        return ([], None, None) if end_to_end else ([], None)

    n_targets = np.asarray(y).shape[1] if np.asarray(y).ndim > 1 else None
    quantity_hat = None

    if prediction_method == 'medoid':
        predictions = return_medoid(y)
    elif prediction_method == 'optimal':
        result = call_optimization_problem(optimization_problem, y, x, initial_solution,
                                           lagrangian_multiplier, verbose,
                                           w=w, quantities=quantities, capacity=capacity)
        if isinstance(result, (tuple, list)) and len(result) == 3:
            # formulation returns (predictions, split_evaluation, quantity_hat):
            # its own objective value is the split evaluation, so use it as is
            predictions, split_evaluation, quantity_hat = result
            return (predictions, split_evaluation, quantity_hat) if end_to_end \
                else (predictions, split_evaluation)
        predictions = as_prediction_vector(result, n_targets)
    elif prediction_method == 'median':
        predictions = return_median(y)
    else:
        predictions = return_mean(y)

    y_eval, predictions_eval = apply_weights(y, predictions, w)

    if evaluation_method == 'mse':
        split_evaluation = calculate_mse(y_eval, predictions_eval)
    elif evaluation_method == 'mad':
        split_evaluation = calculate_mad(y_eval, predictions_eval)
    else:
        split_evaluation = calculate_poisson_deviance(y_eval, predictions_eval)

    return (predictions, split_evaluation, quantity_hat) if end_to_end \
        else (predictions, split_evaluation)
