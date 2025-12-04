import numpy as np
import ot
import pandas as pd
import scanpy as sc
import copy
import time
import zss




# ############################Class
class zssNode(zss.Node):
    def __init__(self, label):
        super(zssNode, self).__init__(label)
        self.children = []

    def addkid_with_order(self, node, before=False):
        if before: 
            self.children.insert(0, node)
        else:
            idx = 0
            while idx < len(self.children):
                if list(node.label)[0] < list(self.children[idx].label)[0]:
                    break
                idx += 1
            self.children.insert(idx, node)
        return self

    def addkid(self, node, before=False):
        if before:  
             self.children.insert(0, node)
        else: 
            self.children.append(node)
        return self

    
class myNode:
    def __init__(self, label):
        self.label = label
        self._parent = None
        self.children = []

    def addkid(self, node):
        if node not in self.children:
            node.parent = self
        return self

    def get_all_descendants_labels(self):
        labels = [self.label]
        for child in self.children:
            labels.extend(child.get_all_descendants_labels())
        return labels

    @property
    def kids(self):
        return [node.label for node in self.children]

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, node):
        self._parent = node
        if self._parent is not None and self not in self._parent.children:
            self._parent.children.append(self)
            
    def find_node(self, label):
        if self.label == label:
            return self
        for child in self.children:
            result = child.find_node(label)
            if result is not None:
                return result
        if label in self.label:
            return self
        for child in self.children:
            result = child.find_node(label)
            if result is not None:
                return result
        return None

    def __repr__(self, level=0):
        ret = "----" * level + repr(self.label) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret
##############################################


def get_metacells(adata, cell_type_col, sample_size=100):
    meta_cell_list = []
    label_list = []

    cell_type_counts = adata.obs[cell_type_col].value_counts()

    for cell_type in cell_type_counts.index:
        cell_indices = adata.obs[adata.obs[cell_type_col] == cell_type].index
        
        if cell_type_counts[cell_type] >= sample_size:
            sampled_idx = np.random.choice(cell_indices, size=sample_size, replace=False)
        else:
            sampled_idx = cell_indices
        
        meta_cells = adata[sampled_idx, :].X
        meta_cell_list.append(meta_cells)
        label_list.extend([cell_type] * len(sampled_idx))

    meta_cell_list = np.vstack(meta_cell_list)  
    label_list = np.array(label_list)
    return meta_cell_list, label_list



def count_confumatrix(adata1, adata2, annotation, transmission_ratio):
    meta_cell1, labels1 = get_metacells(adata1, annotation)
    meta_cell2, labels2 = get_metacells(adata2, annotation)

    M = ot.dist(meta_cell1, meta_cell2)
    M /= M.max()

    a = np.ones((meta_cell1.shape[0],)) / meta_cell1.shape[0]
    b = np.ones((meta_cell2.shape[0],)) / meta_cell2.shape[0]

    df_OT = ot.partial.partial_wasserstein(a, b, M, m = transmission_ratio * min(len(meta_cell1), len(meta_cell2)) / max(len(meta_cell1), len(meta_cell2)))

    df_OT_S2T = pd.DataFrame(np.zeros([len(np.unique(labels1)), len(np.unique(labels2))]), index=np.unique(labels1), columns=np.unique(labels2))
    for cl1 in np.unique(labels1):
        if df_OT[labels1 == cl1, :].sum() > 0:
            for cl2 in np.unique(labels2):
                df_OT_S2T.loc[cl1, cl2] = df_OT[labels1 == cl1, :][:, labels2 == cl2].sum() / df_OT[labels1 == cl1, :].sum()
        else:
            df_OT_S2T.loc[cl1, :] = 0

    df_OT_T2S = pd.DataFrame(np.zeros([len(np.unique(labels2)), len(np.unique(labels1))]), index=np.unique(labels2), columns=np.unique(labels1))
    for cl2 in np.unique(labels2):
        if df_OT[:, labels2 == cl2].sum() > 0:
            for cl1 in np.unique(labels1):
                df_OT_T2S.loc[cl2, cl1] = df_OT[labels1 == cl1, :][:, labels2 == cl2].sum() / df_OT[:, labels2 == cl2].sum()
        else:
            df_OT_T2S.loc[cl2, :] = 0

    return df_OT_T2S, df_OT_S2T










def merge_annotations(adata, annotation, equal):
    for c1, c2 in equal.items():
        adata.obs[annotation] = adata.obs[annotation].replace(c1, c1 + "&" + c2)
        adata.obs[annotation] = adata.obs[annotation].replace(c2, c1 + "&" + c2)
    return adata


def initialize_tree(equal, Parent_kids, unassigned):
    root = myNode("root")
    for c1, c2 in equal.items():
        node = myNode(c1 + "&" + c2)
        root.addkid(node)

    for parent, children in Parent_kids.items():
        parent_node = myNode(parent)
        root.addkid(parent_node)
        for child in children:
            child_node = myNode(child)
            parent_node.addkid(child_node)

    for unassigned_node in unassigned.keys():
        node = myNode(unassigned_node)
        root.addkid(node)
    return root
        
def get_bottom_labels(node):
    if not node.children:
        return [node.label]
    bottom_labels = []
    for child in node.children:
        bottom_labels.extend(get_bottom_labels(child))
    return bottom_labels

def add_equal_anno(root, equal_dict):
    for anno in equal_dict:
        node = root.find_node(anno)
        node.label = anno + "&" + equal_dict[anno]
    return root

def find_parent_labels(root, node_labels):
    label_to_node = {}

    def fill_label_to_node_dict(node):
        label_to_node[node.label] = node
        for child in node.children:
            fill_label_to_node_dict(child)

    fill_label_to_node_dict(root)

    parent_labels = set()

    for label in node_labels:
        if label in label_to_node:
            parent_node = label_to_node[label].parent
            if parent_node.label != "root":
                parent_labels.add(parent_node.label)
            else:
                parent_labels.add(label)

    return list(parent_labels)


def get_key_by_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    



###############################harmonization####################################
def analyze_transmission(D1, D2):
    equal = {}
    Parent_children = {}
    unassigned = {}

    C1 = D1.index
    C2 = D1.columns
    

    # Rule 1: Find equal cell types
    for c1 in C1:
        if (c1 not in Parent_children) and (c1 not in Parent_children.values()):
            for c2 in C2:
                if D1.loc[c1, c2] > 0.6 and D2.loc[c2, c1] > 0.6:
                    equal[c1] = c2
    
    # Rule 2: Find parent-child relationships
    for c1 in C1:
        if (c1 in equal) or (c1 in equal.values()) :
            continue

        non_zero_D1 = (D1.loc[c1, :] > 0).sum()
        child_nodes_D1 = []
        for c2 in C2:
            if c1 in equal and equal[c1] == c2:
                continue
            if D1.loc[c1, c2] > 0 :
                corresponding_row = D2.loc[c2]
                if (corresponding_row[c1] > 0.5):
                    child_nodes_D1.append(c2)
        if len(child_nodes_D1):
            Parent_children[c1] = child_nodes_D1
    
    flat_list = [item for sublist in Parent_children for item in Parent_children[sublist]]


    for c2 in C2:
        if (c2 in equal) or (c2 in equal.values()) or (c2 in flat_list):
            continue

        non_zero_D2 = (D2.loc[c2, :] > 0).sum()
        child_nodes_D2 = []
        for c1 in C1:
            if c2 in equal.values() and list(equal.keys())[list(equal.values()).index(c2)] == c1:
                continue
            if D2.loc[c2, c1] > 0:
                corresponding_row = D1.loc[c1]
                if (corresponding_row[c2] > 0.5):
                    child_nodes_D2.append(c1)
        if len(child_nodes_D2):
            Parent_children[c2] = child_nodes_D2
            


    flat_list = [item for sublist in Parent_children for item in Parent_children[sublist]]
    # Rule 3: Find unassigned cell types
    for c2 in C2:
        found = False
        for c1 in C1:
            if D1.loc[c1, c2] > 0.05:
                found = True
                break

        if (c2 not in equal.values()) and (c2 not in flat_list) and (c2 not in Parent_children):
            unassigned[c2] = 'Not in D1'

    for c1 in C1:
        found = False
        for c2 in C2:
            if D2.loc[c2, c1] > 0.05:
                found = True
                break

        if (c1 not in equal) and (c1 not in flat_list) and (c1 not in Parent_children):
            unassigned[c1] = 'Not in D2'

    return equal, Parent_children, unassigned












def subsequent_comparisons(tree, new_adata, adata, annotation, transmission_ratio, parent_list=[], chilren_record_dict={},
                           failed_assigned=[], first_call=True, failed_call=False, iteration=False):
    """
    Perform subsequent comparisons to harmonize annotations between datasets.

    Parameters:
        tree: The tree structure representing the current annotation hierarchy.
        new_adata: New dataset to be compared.
        adata: Original dataset.
        annotation: Column name for cell type annotations.
        transmission_ratio: Ratio for partial optimal transport.
        parent_list: List of parent nodes to compare (default is empty).
        chilren_record_dict: Dictionary to record child nodes (default is empty).
        failed_assigned: List of annotations that failed to be assigned (default is empty).
        first_call: Flag indicating if this is the first call (default is True).
        failed_call: Flag indicating if this is a retry after a failed assignment (default is False).
        iteration: Flag indicating if this is an iterative call (default is False).

    Returns:
        Updated tree structure.
    """
    if first_call:
        # Initialize variables for the first call
        failed_parent_list = []
        parent_list = []
        chilren_record_dict = {}
        failed_assigned = []
        descendants_labels = get_bottom_labels(tree)
        adata_new0 = adata[adata.obs[annotation].isin(descendants_labels)]
        adata_unassigned = adata[adata.obs[annotation].isin(tree.kids)]
        
        # Calculate partial optimal transport
        df_OT_T2S_new, df_OT_S2T_new = count_confumatrix(adata_unassigned, new_adata, annotation, transmission_ratio)
        equal_new, parent_children_new, unassigned_new = analyze_transmission(df_OT_S2T_new, df_OT_T2S_new)
        
        # List to track unassigned nodes to be removed
        unassigned_to_remove = []

        ### First iteration: Compare with the top-level nodes of the tree.
        ### If a new cell type is found in D2, it is added as a child of the root node.
        ### These new cell types are then removed from the unassigned list.
        for unassigned_node in unassigned_new.keys():
            if unassigned_new[unassigned_node] == 'Not in D2':
                unassigned_to_remove.append(unassigned_node)
            if unassigned_new[unassigned_node] == 'Not in D1':
                tree.addkid(myNode(unassigned_node))
                unassigned_to_remove.append(unassigned_node)

    ### If not the first call, extract the parent nodes for comparison
    elif (not first_call) and (not failed_call):
        adata_new0 = adata[adata.obs[annotation].isin(parent_list)]
        unassigned_to_remove = []
        parent_list = []

    ### If this is a retry after a failed assignment, find parent labels and update the dataset
    elif failed_call:
        unassigned_to_remove = []
        parent_list = find_parent_labels(tree, parent_list)
        adata_new0 = adata[adata.obs[annotation].isin(parent_list)]
        failed_assigned = []

    ### D2_list stores annotations from D2 that acted as parent nodes in the previous round
    D2_list = []

    ### Calculate partial optimal transport for the current comparison
    df_OT_T2S_new, df_OT_S2T_new = count_confumatrix(adata_new0, new_adata, annotation, transmission_ratio)
    equal_new, parent_children_new, unassigned_new = analyze_transmission(df_OT_S2T_new, df_OT_T2S_new)

    ### Handle irrelevant relationships
    ### Remove nodes from D1 that are irrelevant
    for unassigned_node in unassigned_new.keys():
        if unassigned_new[unassigned_node] == 'Not in D2':
            unassigned_to_remove.append(unassigned_node)

    ### Remove irrelevant nodes from the unassigned list
    for unassigned_node in unassigned_to_remove:
        if unassigned_node in unassigned_new:
            unassigned_new.pop(unassigned_node)

    ### If a node is still unassigned after removal, it indicates a failed match.
    ### Add it to the failed_assigned list for further processing.
    for unassigned in unassigned_new:
        if (unassigned not in failed_assigned) and (unassigned_new[unassigned] == 'Not in D1'):
            failed_assigned.append(unassigned)

    ### Handle unassigned nodes at the bottom level.
    ### If a node is equal to another, add it as a child of the equal node.
    equal_to_remove = []
    for cell in equal_new:
        if equal_new[cell] in failed_assigned:
            tree.find_node(cell).addkid(myNode(equal_new[cell]))
            failed_assigned.remove(equal_new[cell])
            equal_to_remove.append(cell)
    for cell in equal_to_remove:
        equal_new.pop(cell)

    ### Correct misclassified equal nodes that are actually parent nodes of previous children
    for parent in chilren_record_dict:
        if (parent in equal_new.values()) and (
                len(chilren_record_dict[parent]) < len(tree.find_node(get_key_by_value(equal_new, parent)).kids)):
            childnode = myNode(parent)
            tree.find_node(get_key_by_value(equal_new, parent)).addkid(childnode)
            for i in chilren_record_dict[parent]:
                previous_parent = tree.find_node(i).parent
                tree.find_node(i).parent = childnode
                previous_parent.children.remove(tree.find_node(i))
            equal_new.pop(get_key_by_value(equal_new, parent))

    ### If this is not a retry after a failed assignment, process equal relationships
    if not failed_call:
        adata = merge_annotations(adata, annotation, equal_new)
        add_equal_anno(tree, equal_new)

    ### Handle parent-child relationships
    for parent, children in parent_children_new.items():
        child_preparent = []

        ### If the child is from D1, store its previous parent
        for child in children:
            if tree.find_node(child) is not None:
                child_preparent.append(tree.find_node(child).parent.label)

        ### If the child is from D1 and its previous parent is not the root node
        for child in children:
            if (tree.find_node(child) is not None) and (tree.find_node(child).parent.label != "root"):
                if tree.find_node(child).parent.label not in parent_list:
                    parent_list.append(tree.find_node(child).parent.label)

                if parent not in D2_list:
                    D2_list.append(parent)
                chilren_record_dict[parent] = children

            ### If the child is from D2, add it as a child of the parent node
            elif tree.find_node(child) is None:
                childnode = myNode(child)
                tree.find_node(parent).addkid(childnode)
                if child in chilren_record_dict:
                    for i in chilren_record_dict[child]:
                        previous_parent = tree.find_node(i).parent
                        tree.find_node(i).parent = childnode
                        previous_parent.children.remove(tree.find_node(i))

            ### If the child's previous parent is the root node
            elif (tree.find_node(child).parent.label == "root") & (all(item == "root" for item in child_preparent)):
                if tree.find_node(parent) is None:
                    childnode = myNode(parent)
                    tree.addkid(childnode)
                else:
                    childnode = tree.find_node(parent)
                tree.find_node(child).parent = childnode
                tree.children.remove(tree.find_node(child))

            ### If the child's previous parent is the root node but not all children have root as parent
            elif (tree.find_node(child).parent.label == "root") & (any(item != "root" for item in child_preparent)):
                parent_list.append(child)

    ### Filter parent_list to include only root-level parents or those not already in the list
    parent_list = [label for label in parent_list if (tree.find_node(label).parent.label == "root" or tree.find_node(label).parent.label not in parent_list)]

    ### Recursively process the parent list if D2_list is not empty and this is not a retry
    if (D2_list != []) & (failed_call == False):
        new_adata1 = new_adata[new_adata.obs[annotation].isin(D2_list)]
        subsequent_comparisons(tree, new_adata1, adata, annotation, transmission_ratio, parent_list, chilren_record_dict, failed_assigned,
                               first_call=False, failed_call=False, iteration=True)

    ### Retry failed assignments if there are any and this is not an iterative call
    if (failed_assigned != []) & (iteration == False):
        if first_call:
            parent_list = get_bottom_labels(tree)
            new_adata = new_adata[new_adata.obs[annotation].isin(failed_assigned)]
            subsequent_comparisons(tree, new_adata, adata, annotation, transmission_ratio, parent_list, chilren_record_dict,
                                   failed_assigned, first_call=False, failed_call=True, iteration=False)
        else:
            new_adata = new_adata[new_adata.obs[annotation].isin(failed_assigned)]
            subsequent_comparisons(tree, new_adata, adata, annotation, transmission_ratio, parent_list, chilren_record_dict,
                                   failed_assigned, first_call=False, failed_call=True, iteration=False)

    return tree
############################################################################################



    
# def do_harmonization(adata, annotation_index, batch_index, sample_size=100, transmission_ratio=0.4, batch_order=None):
#     """
#     Perform automatic comparison and harmonization of all batches in the adata object.

#     Parameters:
#         adata: AnnData object containing single-cell data.
#         annotation_index: Column name for cell type annotations.
#         batch_index: Column name for batch information.
#         sample_size: Number of cells to sample per annotation group (default 100).
#         transmission_ratio: Ratio for partial optimal transport (default 0.4).
#         batch_order: List specifying the order of batches. If None, batches are sorted by annotation count (default None).
#     """
#     start_time = time.time()

#     root = myNode("root")

#     adata.obs[annotation_index] = adata.obs[batch_index].astype(str) + '-' + adata.obs[annotation_index].astype(str)

#     adataori = adata.copy()
#     adata = adata.copy()

#     batches = adata.obs[batch_index].unique()
#     batch_counts = []

#     for batch in batches:
#         batch_data = adata[adata.obs[batch_index] == batch]
#         annotation_counts = batch_data.obs[annotation_index].value_counts().sum()
#         batch_counts.append((batch, annotation_counts))

#     if batch_order is None:
#         batch_counts.sort(key=lambda x: x[1], reverse=True)
#     else:
#         if not isinstance(batch_order, list):
#             raise ValueError("batch_order should be a list. Please provide a list of batch orders.")
#         batch_counts.sort(key=lambda x: batch_order.index(x[0]) if x[0] in batch_order else len(batch_order))

#     batch_data_dict = {}
#     for i, (batch, _) in enumerate(batch_counts):
#         batch_data_dict[f"adata{i}"] = adata[adata.obs[batch_index] == batch]

#     batch_names = list(batch_data_dict.keys())

#     for i in range(len(batch_names)):
#         if i == 0:
#             adata0 = batch_data_dict[batch_names[i]]
#             batch_name_0 = batch_counts[i][0]  # Extract batch name
#         elif i == 1:
#             adata1 = batch_data_dict[batch_names[i]]
#             batch_name_1 = batch_counts[i][0]  # Extract batch name
#             print(f"Initializing tree from {batch_name_0} and {batch_name_1}🌲")

#             df_OT_T2S, df_OT_S2T = count_confumatrix(adata0, adata1, annotation_index, transmission_ratio)
#             equal, Parent_kids, unassigned = analyze_transmission(df_OT_S2T, df_OT_T2S)

#             adataori = merge_annotations(adataori, annotation_index, equal)
#             root = initialize_tree(equal, Parent_kids, unassigned)
#         else:
#             adata_subs = batch_data_dict[batch_names[i]]
#             print(f"Adding annotations from {batch_counts[i][0]}🍃")
#             subsequent_comparisons(root, adata_subs, adataori, annotation_index, transmission_ratio)

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Batch comparison completed. Time elapsed: {elapsed_time:.4f} seconds")

#     return root



def do_harmonization(adata, annotation_index, batch_index, sample_size=100, transmission_ratio=0.4, batch_order=None):
    """
    Perform automatic comparison and harmonization of all batches in the adata object.

    Parameters:
        adata: AnnData object containing single-cell data.
        annotation_index: Column name for cell type annotations.
        batch_index: Column name for batch information.
        sample_size: Number of cells to sample per annotation group (default 100).  # kept for API consistency
        transmission_ratio: Ratio for partial optimal transport (default 0.4).
        batch_order: List specifying the order of batches. If None, batches are sorted by
                     (global Leiden cluster-count / annotation-count) per batch.
    """
    start_time = time.time()
    root = myNode("root")

    # Make annotation labels unique per-batch (keeps your original convention)
    adata = adata.copy()
    adata.obs[annotation_index] = adata.obs[batch_index].astype(str) + '-' + adata.obs[annotation_index].astype(str)
    adataori = adata.copy()

    # ---------- Decide batch order ----------
    batches = adata.obs[batch_index].astype(str).unique().tolist()
    if batch_order is None:
        # One-shot global Leiden, then order batches by (n_leiden_clusters / n_annotations) descending
        _ad = adata.copy()

        # === 自适应 PCA 设定，避免 n_components > min(n_samples, n_features) 报错 ===
        # 用 sklearn 的 'auto' 求解器更稳；n_pcs 取 min(50, n_obs-1, n_vars)，且至少为 2
        n_obs = _ad.n_obs
        n_vars = _ad.n_vars
        n_pcs = max(2, min(50, n_obs - 1, n_vars))

        sc.tl.pca(_ad, n_comps=n_pcs, random_state=0, use_highly_variable=False, svd_solver="auto")
        n_pcs_used = min(n_pcs, _ad.obsm["X_pca"].shape[1])
        sc.pp.neighbors(_ad, n_neighbors=15, n_pcs=n_pcs_used, random_state=0)
        sc.tl.leiden(_ad, resolution=1.0, key_added="leiden", random_state=0)

        rows = []
        for b in batches:
            sub = _ad[_ad.obs[batch_index].astype(str) == b]
            n_leiden = int(sub.obs["leiden"].nunique()) if sub.n_obs > 0 else 0
            n_anno   = int(sub.obs[annotation_index].nunique()) if sub.n_obs > 0 else 0
            score    = n_leiden / max(n_anno, 1)
            rows.append((b, n_leiden, n_anno, score))
        rows.sort(key=lambda x: x[3], reverse=True)
        ordered_batches = [r[0] for r in rows]
        print(f"[Global-Leiden order] resolution=1.0; PCA n_pcs={n_pcs}; score = n_leiden / n_annotations")
        print(pd.DataFrame(rows, columns=["batch", "n_leiden", "n_annotations", "score"])
                .sort_values("score", ascending=False))
    else:
        # Respect user-specified order; filter to existing batches only
        ordered_batches = [str(b) for b in batch_order if str(b) in set(batches)]

    # Build dict in the chosen order
    batch_data_dict = {}
    for i, b in enumerate(ordered_batches):
        batch_data_dict[f"adata{i}"] = adata[adata.obs[batch_index].astype(str) == b]

    batch_names = list(batch_data_dict.keys())

    # ---------- Progressive harmonization ----------
    for i in range(len(batch_names)):
        if i == 0:
            adata0 = batch_data_dict[batch_names[i]]
            batch_name_0 = ordered_batches[i]
        elif i == 1:
            adata1 = batch_data_dict[batch_names[i]]
            batch_name_1 = ordered_batches[i]
            print(f"Initializing tree from {batch_name_0} and {batch_name_1}🌲")

            # 如果你的 count_confumatrix 不接受 random_state 参数，去掉 random_state=0 即可
            df_OT_T2S, df_OT_S2T = count_confumatrix(adata0, adata1, annotation_index, transmission_ratio)
            equal, Parent_kids, unassigned = analyze_transmission(df_OT_S2T, df_OT_T2S)

            adataori = merge_annotations(adataori, annotation_index, equal)
            root = initialize_tree(equal, Parent_kids, unassigned)
        else:
            adata_subs = batch_data_dict[batch_names[i]]
            print(f"Adding annotations from {ordered_batches[i]}🍃")
            # 同理：若 subsequent_comparisons 不支持 random_state，去掉该参数
            subsequent_comparisons(
                root, adata_subs, adataori, annotation_index, transmission_ratio
            )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Batch comparison completed. Time elapsed: {elapsed_time:.4f} seconds")

    return root






# 新增：一次全局 Leiden 后按 ratio 排序
def _order_batches_by_global_leiden(adata, batch_index, annotation_index,
                                    n_pcs=50, n_neighbors=15, resolution=1.0,
                                    key_added="leiden", random_state=0):
    _ad = adata.copy()
    sc.tl.pca(_ad, n_comps=n_pcs, random_state=random_state, use_highly_variable=False, svd_solver="arpack")
    sc.pp.neighbors(_ad, n_neighbors=n_neighbors, n_pcs=min(n_pcs, _ad.obsm["X_pca"].shape[1]), random_state=random_state)
    sc.tl.leiden(_ad, resolution=resolution, key_added=key_added, random_state=random_state)

    rows = []
    for b in _ad.obs[batch_index].astype(str).unique():
        sub = _ad[_ad.obs[batch_index].astype(str) == b]
        n_leiden = int(sub.obs[key_added].nunique()) if sub.n_obs > 0 else 0
        n_anno   = int(sub.obs[annotation_index].nunique()) if sub.n_obs > 0 else 0
        score    = n_leiden / max(n_anno, 1)
        rows.append((b, n_leiden, n_anno, score))
    rows.sort(key=lambda x: x[3], reverse=True)
    return [r[0] for r in rows], pd.DataFrame(rows, columns=["batch", "n_leiden", "n_annotations", "score"])




# def do_harmonization(adata, annotation_index, batch_index, sample_size=100,
#                      transmission_ratio=0.4, batch_order=None,
#                      alpha: float = 1.0, beta: float = 1.0,
#                      use_log: bool = True, tie_break: str = "cells"):
#     """
#     Perform automatic comparison and harmonization of all batches in the adata object.

#     默认情况下，按注释数+细胞数的先验排序（粗且大优先）；
#     若提供 batch_order 则按指定顺序。
#     -----------------------------------------------------------
#     排序规则：
#         Score(b) = beta * log10(C_b) - alpha * log(N_b)
#         - 注释数 N_b 越少越靠前（惩罚 log N）
#         - 细胞数 C_b 越多越靠前（奖励 log C）
#     参数:
#         adata: AnnData 对象
#         annotation_index: 注释列名
#         batch_index: 批次列名
#         sample_size: 每类抽样数
#         transmission_ratio: OT 传输比例
#         batch_order: 可选；若给出则跳过先验排序
#         alpha, beta, use_log, tie_break: 控制先验排序行为
#     -----------------------------------------------------------
#     """
#     start_time = time.time()

#     root = myNode("root")
#     adata.obs[annotation_index] = (
#         adata.obs[batch_index].astype(str)
#         + '-' + adata.obs[annotation_index].astype(str)
#     )

#     adataori = adata.copy()
#     adata = adata.copy()

#     batches = adata.obs[batch_index].unique()
#     batch_counts = []

#     # =====================================================
#     # ① 若未提供 batch_order，自动根据 N_b+C_b 排序
#     # =====================================================
#     if batch_order is None:
#         print("🔸 未指定 batch_order，将根据注释数量与细胞数量自动排序...\n")
#         rows = []
#         for batch in batches:
#             batch_data = adata[adata.obs[batch_index] == batch]
#             n_cells = int(batch_data.n_obs)
#             n_annos = int(batch_data.obs[annotation_index].nunique())
#             C_safe = max(n_cells, 1)
#             N_safe = max(n_annos, 1)
#             if use_log:
#                 score = beta * np.log10(C_safe) - alpha * np.log(N_safe)
#             else:
#                 score = (C_safe ** beta) / (N_safe ** alpha)
#             rows.append({"batch": batch, "n_cells": n_cells, "n_annos": n_annos, "score": score})
#         table = pd.DataFrame(rows).set_index("batch")
#         if tie_break == "cells":
#             table = table.sort_values(by=["score", "n_cells", "n_annos"], ascending=[False, False, True])
#         elif tie_break == "annos":
#             table = table.sort_values(by=["score", "n_annos", "n_cells"], ascending=[False, True, False])
#         else:
#             table = table.sort_values(by=["score"], ascending=[False])
#         batch_order = table.index.tolist()

#         print("自动排序结果（粗且大优先）:")
#         print(table[["n_cells", "n_annos", "score"]].round(4).to_string())
#         print("\n实际对齐顺序：", " → ".join(batch_order), "\n")
#     else:
#         if not isinstance(batch_order, list):
#             raise ValueError("batch_order 必须为 list。")

#     # =====================================================
#     # ② 构建批次数据集
#     # =====================================================
#     batch_data_dict = {}
#     for batch in batch_order:
#         batch_data_dict[batch] = adata[adata.obs[batch_index] == batch]

#     batch_names = list(batch_data_dict.keys())

#     # =====================================================
#     # ③ 执行逐批对齐逻辑
#     # =====================================================
#     for i in range(len(batch_names)):
#         if i == 0:
#             adata0 = batch_data_dict[batch_names[i]]
#             batch_name_0 = batch_names[i]
#         elif i == 1:
#             adata1 = batch_data_dict[batch_names[i]]
#             batch_name_1 = batch_names[i]
#             print(f"Initializing tree from {batch_name_0} and {batch_name_1} 🌲")

#             df_OT_T2S, df_OT_S2T = count_confumatrix(adata0, adata1, annotation_index, transmission_ratio)
#             equal, Parent_kids, unassigned = analyze_transmission(df_OT_S2T, df_OT_T2S)

#             adataori = merge_annotations(adataori, annotation_index, equal)
#             root = initialize_tree(equal, Parent_kids, unassigned)
#         else:
#             adata_subs = batch_data_dict[batch_names[i]]
#             print(f"Adding annotations from {batch_names[i]} 🍃")
#             subsequent_comparisons(root, adata_subs, adataori, annotation_index, transmission_ratio)

#     # =====================================================
#     # ④ 输出总耗时
#     # =====================================================
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"\n✅ Batch comparison completed. Time elapsed: {elapsed_time:.2f} seconds.")

#     return root



