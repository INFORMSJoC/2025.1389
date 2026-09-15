# Based on the work by baydoganm/mtTrees
import sys
import os

import time
import inspect
import numpy as np
import pandas as pd
from sklearn import preprocessing
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from SingleDepth_ORT import SingleDepthMIP

class Node:
    """
    Node class for a Decision Tree.

    Attributes:
        right (Node): Right child node.
        left (Node): Left child node.
        column (int): Index of the feature used for splitting.
        column_name (str): Name of the feature.
        threshold (float): Threshold for the feature split.
        id (int): Identifier for the node.
        depth (int): Depth of the node in the tree.
        is_terminal (bool): Indicates if the node is a terminal node.
        prediction (numpy.ndarray): Predicted values for the node.
        count (int): Number of samples in the node.

    Methods:
        No specific methods are defined in this class.

        It will be mainly used in the construction of the tree.
    """
    def __init__(self):
        self.right = None
        self.left = None
        self.column = None
        self.column_name = None
        self.threshold = None
        self.id = None
        self.depth = None
        self.is_terminal = False
        self.prediction = None
        self.quantities = None
        self.count = None
        self.parent_mse = None

class OCRT:
    """
    Predictive Clustering Tree.

    Args:
        max_depth (int): Maximum depth of the tree.
        min_samples_leaf (int): Minimum number of samples in a leaf node.
        min_samples_split (int): Minimum number of samples to split a node.
        w: Optional weight/parameter object forwarded to split_criteria and
            leaf_prediction_method. Leave as None for criteria functions that
            do not take it.
        pass_w (bool or None): Controls whether w is forwarded.
            None (default) -> decided automatically, see _resolve_pass_w.
            True  -> always call fn(labels, features, w, nof_infeas, best_sol).
            False -> always call fn(labels, features, nof_infeas, best_sol).
        split_style (str): Splitting style.
        verbose (bool): Whether to print verbose information.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_samples_leaf (int): Minimum number of samples in a leaf node.
        min_samples_split (int): Minimum number of samples to split a node.
        split_style (str): Splitting style (e.g. 'custom')
        verbose (bool): Whether to print verbose information.
        Tree (Node): Root node of the predictive clustering tree.

    Methods:
        buildDT(features, labels, node):
            Build the predictive clustering tree.

        fit(features, labels):
            Fit the predictive clustering tree to the data.

        nodePredictions(y):
            Calculate predictions for a node.

        applySample(features, depth, node):
            Passes one object through the decision tree and returns the prediction.

        apply(features, depth):
            Returns the node id for each X.

        get_rules(features, depth, node, rules):
            Returns the decision rules for feature selection.

        calcBestSplit(features, labels, current_label):
            Calculates the best split based on features and labels.

        calcBestSplitCustom(features, labels):
            Calculates the best custom split for features and labels.
    """
    def __init__(self, max_depth = 5,
                 min_samples_leaf = 5,
                 min_samples_split = 10,
                 w = None,
                 quantities = None,
                 capacity = None,
                 split_criteria = None,
                 leaf_prediction_method = None,
                 nof_infeasibilities_method = None,
                 verbose = False,
                 use_hashmaps = True,
                 use_initial_solution = True,
                 bforce=True,
                 problem=None,
                 pass_w=None,
                 pass_quantities=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.split_criteria = split_criteria
        self.nof_infeasibilities_method = nof_infeasibilities_method
        self.leaf_prediction_method = leaf_prediction_method
        self.verbose = verbose
        self.Tree = None
        self.use_hashmaps = use_hashmaps
        self.use_initial_solution = use_initial_solution
        self.bforce = bforce
        self.problem = problem
        self.all_leaf_nodes = []
        self.w = w
        self.quantities = quantities
        self.capacity = capacity
        self.pass_w = pass_w
        self.pass_quantities = pass_quantities
        self._pass_w_cache = {}

    # -----------------------------------------------------------------
    # Compatibility layer. Criteria functions come in three flavours:
    #   (a) fn(labels, features, nof_infeas, best_solution)
    #   (b) fn(labels, features, w, nof_infeas, best_solution)
    #   (c) fn(labels, features, w, quantities, capacity, nof_infeas,
    #          best_solution)                       <- end-to-end experiments
    # Flavour (c) may also return a third value, the optimised quantities.
    # -----------------------------------------------------------------
    def _uses_quantities(self):
        """
        Whether this tree runs in end-to-end (quantity) mode.

        Explicit pass_quantities wins; otherwise it is on exactly when a
        quantities frame was supplied.
        """
        if self.pass_quantities is not None:
            return bool(self.pass_quantities)
        return self.quantities is not None

    def _resolve_pass_w(self, fn):
        """
        Decide whether fn expects the extra w argument.

        Order of precedence:
          1. explicit self.pass_w (True/False) set by the user
          2. signature inspection (5 accepted positional arguments -> flavour b)
          3. fallback: pass w only if a non-None w was provided
        """
        if self.pass_w is not None:
            return bool(self.pass_w)
        if fn in self._pass_w_cache:          # signature inspection is cached:
            return self._pass_w_cache[fn]     # this runs inside the split loop
        try:
            params = [
                p for p in inspect.signature(fn).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if any(p.kind == p.VAR_POSITIONAL
                   for p in inspect.signature(fn).parameters.values()):
                raise ValueError
            decision = len(params) >= 5
        except (TypeError, ValueError):
            decision = self.w is not None
        self._pass_w_cache[fn] = decision
        return decision

    @staticmethod
    def _unpack_criteria(result):
        """Normalise a 2- or 3-value criteria return to (prediction, perf, quantity)."""
        if isinstance(result, (tuple, list)) and len(result) == 3:
            return result[0], result[1], result[2]
        prediction, perf = result
        return prediction, perf, None

    def _call_criteria(self, fn, labels, features, node_quantities=None):
        """
        Invoke a criteria/leaf-prediction function with the right signature.

        Always returns the triple (prediction, perf, quantity); quantity is
        None for the flavours that do not produce one.
        """
        if self._uses_quantities():
            return self._unpack_criteria(
                fn(labels, features, self.w, node_quantities, self.capacity,
                   self.nof_infeasibilities_method, self.best_solution))
        if self._resolve_pass_w(fn):
            return self._unpack_criteria(
                fn(labels, features, self.w,
                   self.nof_infeasibilities_method, self.best_solution))
        return self._unpack_criteria(
            fn(labels, features,
               self.nof_infeasibilities_method, self.best_solution))

    def _node_quantities(self, index):
        """Rows of the quantities frame belonging to the given label index."""
        if not self._uses_quantities() or self.quantities is None:
            return None
        return self.quantities.loc[index].to_numpy()

    def buildDT(self, features, labels, node,pred=None,quant=None):
        """
        Build the predictive clustering tree.

        Args:
            features (pandas.DataFrame): The input features used for building the tree.
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.
            node (Node): The current node in the tree being built.
        """
        # print('YYY',node.id)
        # `pred is not None` rather than `if pred:` so numpy arrays are safe;
        # an empty container still falls through to recomputation, as before.
        if pred is not None and not (hasattr(pred, '__len__') and len(pred) == 0):
            node.prediction = pred
        else:
            node.prediction, _, node.quantities = self._call_criteria(
                self.split_criteria, labels.to_numpy(), features.to_numpy(),
                self._node_quantities(labels.index))

        # quantities optimised by the parent split for this specific child win
        if quant is not None:
            node.quantities = quant
        # print(node.prediction)
        node.count = labels.shape[0]
        if node.depth >= self.max_depth:
            node.is_terminal = True
            self.all_leaf_nodes.append(node)
            return

        if features.shape[0] < self.min_samples_split:
            node.is_terminal = True
            self.all_leaf_nodes.append(node)
            return
        
        current_label = range(labels.shape[1])
        target = labels
        # print('XXX',node.id,features.shape)
        best_children_quants = (None, None)
        if self.bforce:
            # print('HAHAHAHHA!!!!')
            split_info, split_gain, n_cuts, split_quantities = self.calcBestSplitCustom(features, target)
            if n_cuts == 0:
                node.is_terminal = True
                self.all_leaf_nodes.append(node)
                return
            min_max_scaler = preprocessing.MinMaxScaler()
            split_gain_scaled_total = min_max_scaler.fit_transform(split_gain)[:, 0]
            mean_rank_sort = np.argsort(split_gain_scaled_total)
            best_idx = mean_rank_sort[0]
            splitCol = int(split_info[best_idx, 0])
            thresh = split_info[best_idx, 1]
            if split_quantities:
                best_children_quants = split_quantities[best_idx]
        else:
            '''
            split_info, split_gain, n_cuts = self.calcBestSplitCustom(features, target,flag=False)
            if n_cuts == 0:
                node.is_terminal = True
                self.all_leaf_nodes.append(node)
                return
            min_max_scaler = preprocessing.MinMaxScaler()
            split_gain_scaled_total = min_max_scaler.fit_transform(split_gain)[:, 0]
            mean_rank_sort = np.argsort(split_gain_scaled_total)
            splitCol = int(split_info[mean_rank_sort[0], 0])
            thresh = split_info[mean_rank_sort[0], 1]
            '''
            # This must be used when single-depth MIP is used
            if self.use_initial_solution:
                splitCol,thresh,predictions,mse1,mse2,tot_mse = SingleDepthMIP(features,target,self.min_samples_leaf,len(target.columns),self.problem,node.prediction,Tmax=120)
                if node.parent_mse < tot_mse:
                    node.is_terminal = True
                    self.all_leaf_nodes.append(node)
                    return
                if not splitCol:
                    node.is_terminal = True
                    self.all_leaf_nodes.append(node)
                    return
            else:
                # print('HEREEEE',node.id)
                splitCol,thresh,predictions,mse1,mse2,tot_mse = SingleDepthMIP(features,target,self.min_samples_leaf,len(target.columns),self.problem,[],Tmax=120)
                # print('XXXX',splitCol,thresh,predictions,mse1,mse2,tot_mse)
                if node.parent_mse < tot_mse:
                    node.is_terminal = True
                    self.all_leaf_nodes.append(node)
                    return
                if not splitCol:
                    node.is_terminal = True
                    self.all_leaf_nodes.append(node)
                    return
            if predictions == None:
                predictions = [node.prediction,[]]
                node.is_terminal = True
                self.all_leaf_nodes.append(node)
                return
            splitCol-=1
            
            
        # print(node.id,splitCol,thresh)
        node.column = splitCol
        node.column_name = features.columns[splitCol]
        node.threshold = thresh
        labels_left = labels.loc[features.iloc[:,splitCol] <= thresh, :]
        labels_right = labels.loc[features.iloc[:,splitCol] > thresh, :]
        features_left = features.loc[features.iloc[:,splitCol] <= thresh]
        features_right = features.loc[features.iloc[:,splitCol] > thresh]
        # print('WQWQQ',node.id,features_left.shape,features_right.shape)
        # creating left and right child nodes
        
        node.left = Node()
        node.left.depth = node.depth + 1
        node.left.id = 2 * node.id

        node.right = Node()
        node.right.depth = node.depth + 1
        node.right.id = 2 * node.id + 1
        
        if self.bforce:
            # splitting recursively (quantities optimised for each child, if any)
            self.buildDT(features_left, labels_left, node.left, quant=best_children_quants[0])
            self.buildDT(features_right, labels_right, node.right, quant=best_children_quants[1])
        else:
            # splitting recursively
            
            # This must be used when single-depth MIP is used!!!
            node.right.parent_mse = mse2
            node.left.parent_mse = mse1
            self.buildDT(features_left, labels_left, node.left,predictions[0])
            self.buildDT(features_right, labels_right, node.right,predictions[1])
            '''
            self.buildDT(features_left, labels_left, node.left)
            self.buildDT(features_right, labels_right, node.right) 
            '''


    def fit(self, features, labels):
        """
        Fit the predictive clustering tree to the data.

        Args:
            features (pandas.DataFrame): The input features used for building the tree.
            labels (pandas.DataFrame): The labels or target variables corresponding to the features.
        """
        start = time.time()
        self.features = features
        self.labels = labels
        self.preds_dict = {}
        self.Tree = Node()
        self.Tree.depth = 0
        self.Tree.id = 1
        self.Tree.parent_mse = 1e20
        if len(self.labels.shape) > 1:
            self.best_solution = np.zeros(self.labels.shape[1])
        else:
            self.best_solution = np.zeros(1)
        self.best_solution_perf = float('inf')
        self.buildDT(features, labels, self.Tree)
        
        leaf_predictions = {}
        leaf_quantities = {}
        if self.bforce:
            leaves, _ = self.apply(features)
            for leaf_id in np.unique(leaves):
                leaf_indices = np.where(leaves == leaf_id)[0]
                leaf_labels = labels.iloc[leaf_indices].to_numpy()
                leaf_features = features.iloc[leaf_indices].to_numpy()
                leaf_node_quantities = None
                if self._uses_quantities() and self.quantities is not None:
                    leaf_node_quantities = self.quantities.iloc[leaf_indices].to_numpy()
                leaf_predictions[leaf_id], _, leaf_quantities[leaf_id] = self._call_criteria(
                    self.leaf_prediction_method, leaf_labels, leaf_features, leaf_node_quantities)
        else:
            leaves,preds = self.apply(features)
            for leaf_id in np.unique(leaves):
                leaf_indices = np.where(leaves == leaf_id)[0]
                leaf_predictions[leaf_id] = preds[leaf_indices[0]]
                leaf_quantities[leaf_id] = None
            
        self.leaf_predictions_df = pd.DataFrame(leaf_predictions)
        if any(q is not None for q in leaf_quantities.values()):
            self.leaf_quantities_df = pd.DataFrame(
                {k: v for k, v in leaf_quantities.items() if v is not None})
        else:
            self.leaf_quantities_df = None
        end = time.time()
        self.training_duration = end-start

    def predict(self, features):
        '''
        Returns the labels for each X.

        In end-to-end (quantity) mode the return value is the pair
        (predictions, quantities); otherwise it is the predictions array, as
        in the original implementation.
        '''
        
        leaves, _ = self.apply(features)
        predictions = np.asarray(self.leaf_predictions_df[leaves].T)

        if self._uses_quantities():
            if self.leaf_quantities_df is None or self.leaf_quantities_df.empty:
                return predictions, None
            return predictions, np.asarray(self.leaf_quantities_df[leaves].T)

        return predictions

    def predictSample(self, features, depth, node):
        '''
        Passes one object through decision tree and return the probability of it to belong to each class
        '''

        # if we have reached the terminal node of the tree
        if node.is_terminal:
            return node.prediction

        # if we have reached the provided depth
        if node.depth == depth:
            return node.prediction

        if features.iloc[node.column] > node.threshold:
            predicted = self.predictSample(features, depth, node.right)
        else:
            predicted = self.predictSample(features, depth, node.left)

        return predicted

    def applySample(self, features, depth, node):
        """
        Passes one object through the predictive clustering tree and returns the leaf ID.

        Args:
            features (pandas.Series): The input features for a single object.
            depth (int): The depth at which to stop traversing the tree.
            node (Node): The current node in the tree being traversed.

        Returns:
            predicted (int): The predicted node ID.
        """

        # if we have reached the terminal node of the tree
        if node.is_terminal:
            if self.bforce:
                return node.id,None
            else:
                return node.id,node.prediction

        # if we have reached the provided depth
        if node.depth == depth:
            if self.bforce:
                return node.id,None
            else:
                return node.id,node.prediction

        if features.iloc[node.column] > node.threshold:
            predicted = self.applySample(features, depth, node.right)
        else:
            predicted = self.applySample(features, depth, node.left)

        return predicted

    def apply(self, features):
        """
        Returns the node ID for each input object.

        Args:
            features (pandas.DataFrame): The input features for multiple objects.

        Returns:
            predicted_ids (numpy.ndarray): The predicted node IDs for each input object.
        """
        predicted_ids = [self.applySample(features.loc[i], self.max_depth, self.Tree)[0] for i in features.index]
        predicted_ids = np.asarray(predicted_ids)
        if self.bforce: 
            return predicted_ids,None
        else:
            preds = [self.applySample(features.loc[i], self.max_depth, self.Tree)[1] for i in features.index]
            preds = np.asarray(preds)
            return predicted_ids,preds

    def get_rules(self, features, depth, node, rules):
        """
        Returns the decision rules for leaf node assignment.

        Args:
            features (pandas.Series): The input features for a single object.
            depth (int): The depth at which to stop traversing the tree.
            node (Node): The current node in the tree being traversed.
            rules (list): A list to store the decision rules.

        Returns:
            rules (list): The updated list of decision rules.
        """
        # if we have reached the terminal node of the tree
        if node.is_terminal:
            msg = f'Ended at terminal node with ID: {node.id}'
            print(msg)
            return rules

        # if we have reached the provided depth
        if node.depth == depth:
            msg = f'Ended at depth' + str(node.depth)
            print(msg)
            return rules

        if features.iloc[:,node.column].values[0] > node.threshold:
            msg = f'Going right: Node ID: {node.id}, Rule: {features.columns[node.column]} > {node.threshold}'
            print(msg)
            rules.append({features.columns[node.column]: {'min': node.threshold}})
            rules = self.get_rules(features, depth, node.right, rules)
        else:
            msg = f'Going left: Node ID: {node.id}, Rule: {features.columns[node.column]} <= {node.threshold}'
            print(msg)
            rules.append({features.columns[node.column]: {'max': node.threshold}})
            rules = self.get_rules(features, depth, node.left, rules)

        return rules

    def calcBestSplitCustom(self, features, labels,flag=True):
        evaluated_thresholds = {feat: [] for feat in range(features.shape[1])}
        n = features.shape[0]
        cut_id = 0
        n_obj = 1
        split_perf = np.zeros((n * features.shape[1], n_obj))
        split_info = np.zeros((n * features.shape[1], 2))
        # (left_quantity, right_quantity) per cut; entries stay None outside
        # end-to-end mode
        split_quantities = [None] * (n * features.shape[1])
        node_quantities = self._node_quantities(features.index)
        for k in range(features.shape[1]):
            if self.verbose:
                print(f'Feature Index: {k}')
            x = features.iloc[:, k].to_numpy()
            y = labels.to_numpy()
            sort_idx = np.argsort(x)
            sort_x = x[sort_idx]
            sort_y = y[sort_idx, :]
            sort_q = node_quantities[sort_idx, :] if node_quantities is not None else None

            for i in range(self.min_samples_leaf, n - self.min_samples_leaf + 1):
                xi = sort_x[i]
                prev_val = sort_x[i - 1]
                
                # -------------------------------------------------------------
                # HT Changed this part to avoid unnecessary attempts for same thresholds
                if ((xi == sort_x[i - 1]) or (xi in evaluated_thresholds[k])):
                    continue
                # -------------------------------------------------------------

                left_yi = sort_y[:i, :]
                right_yi = sort_y[i:, :]

                left_qi = sort_q[:i, :] if sort_q is not None else None
                right_qi = sort_q[i:, :] if sort_q is not None else None

                if self.use_hashmaps:
                    left_idx = tuple(sorted(features.iloc[sort_idx[:i]].index))
                    right_idx = tuple(sorted(features.iloc[sort_idx[i:]].index))
                    
                    left_xi = features.to_numpy()[sort_idx[:i]]
                    right_xi = features.to_numpy()[sort_idx[i:]]

                    if left_idx not in self.preds_dict:
                        left_prediction, left_perf, left_quant = self._call_criteria(self.split_criteria, left_yi, left_xi, left_qi)
                        self.preds_dict[left_idx] = {'preds': left_prediction, 'perf': left_perf, 'quant': left_quant}
                    else:
                        cached = self.preds_dict[left_idx]
                        left_prediction, left_perf, left_quant = cached['preds'], cached['perf'], cached.get('quant')

                    if right_idx not in self.preds_dict:
                        right_prediction, right_perf, right_quant = self._call_criteria(self.split_criteria, right_yi, right_xi, right_qi)
                        self.preds_dict[right_idx] = {'preds': right_prediction, 'perf': right_perf, 'quant': right_quant}
                    else:
                        cached = self.preds_dict[right_idx]
                        right_prediction, right_perf, right_quant = cached['preds'], cached['perf'], cached.get('quant')
                else:
                    left_xi = features.to_numpy()[sort_idx][:i]
                    right_xi = features.to_numpy()[sort_idx][i:]

                    left_prediction, left_perf, left_quant = self._call_criteria(self.split_criteria, left_yi, left_xi, left_qi)
                    right_prediction, right_perf, right_quant = self._call_criteria(self.split_criteria, right_yi, right_xi, right_qi)

                if self.use_initial_solution:
                    if left_perf < self.best_solution_perf:
                        self.best_solution_perf = left_perf
                        self.best_solution = left_prediction
                    if right_perf < self.best_solution_perf:
                        self.best_solution_perf = right_perf
                        self.best_solution = right_prediction

                left_instance_count = left_yi.shape[0]
                right_instance_count = right_yi.shape[0]
                
                if flag:
                    curr_score = (left_perf * left_instance_count + right_perf * right_instance_count) / n
                else:
                    # DON'T FORGET TO DISABLE split_criteria_with_methods in utils.py: if not bforce: return [], None
                    curr_score = left_perf * left_instance_count + right_perf * right_instance_count
                threshold_val = (xi + prev_val) / 2
                
                if ((xi == sort_x[i - 1]) or (xi in evaluated_thresholds[k])):
                    continue
                else:
                    evaluated_thresholds[k].append(xi)

                split_perf[cut_id, 0] = curr_score
                split_info[cut_id, 0] = k
                split_quantities[cut_id] = (left_quant, right_quant)
                # split_info[cut_id, 1] = xi

                # if i < self.min_samples_leaf or xi == sort_x[i + 1]:
                #     continue
                split_info[cut_id, 1] = threshold_val
                cut_id += 1

        split_info = split_info[range(cut_id), :]
        split_gain = split_perf[range(cut_id), :]
        split_quantities = split_quantities[:cut_id]
        n_cuts = cut_id

        # filter NaN gains, keeping split_quantities aligned with the arrays
        valid_mask = ~np.isnan(split_gain).any(axis=1)
        split_info = split_info[valid_mask,:]
        split_gain = split_gain[valid_mask,:]
        split_quantities = [q for q, valid in zip(split_quantities, valid_mask) if valid]

        return split_info, split_gain, n_cuts, split_quantities
