
import numpy as np
import ot
import pandas as pd
import scanpy as sc
import copy
import time
import zss

# ======================= 新增：顺序体检辅助函数（只用到 1/2/3 三个指标） =======================
def _pairwise_similarity(adata_i, adata_j, annotation, transmission_ratio=0.4):
    """
    使用你现有的 count_confumatrix + analyze_transmission 计算 i,j 的相似度。
    相似度定义（简洁稳妥）：(等价对数量 + 父子对数量) / 平均类别数
    """
    df_T2S, df_S2T = count_confumatrix(adata_i, adata_j, annotation, transmission_ratio)
    equal, parent_children, _ = analyze_transmission(df_S2T, df_T2S)
    match_cnt = len(equal) + sum(len(v) for v in parent_children.values())
    n_i = adata_i.obs[annotation].nunique()
    n_j = adata_j.obs[annotation].nunique()
    denom = max(1, int((n_i + n_j) / 2))
    return float(match_cnt) / denom

def _build_pairwise_S(batch_dict, annotation, transmission_ratio=0.4):
    """
    为所有 batch 构建相似度矩阵 S（对称）
    batch_dict: {name: AnnData(原始注释列)}
    """
    names = list(batch_dict.keys())
    n = len(names)
    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            S[i, j] = S[j, i] = _pairwise_similarity(batch_dict[names[i]], batch_dict[names[j]],
                                                     annotation, transmission_ratio)
    return names, S

def _sim_to_pool(S, order_idx, step, pool_mask):
    """
    计算当前 step 将要加入的索引 k 对已加入池的平均相似度
    """
    k = order_idx[step]
    pool_indices = [order_idx[t] for t in range(step) if pool_mask[order_idx[t]]]
    if not pool_indices:
        return 0.0
    return float(np.mean([S[k, j] for j in pool_indices]))

def _top2_gap(S, order_idx, step, used_mask):
    """
    计算当前候选与“尚未加入”的其它候选之间的 top2 差值
    """
    k = order_idx[step]
    remaining = [order_idx[t] for t in range(step, len(order_idx)) if not used_mask[order_idx[t]] or order_idx[t]==k]
    # 这里的 remaining 包含当前 k 自身，但我们会排掉
    cand = [S[k, j] for j in remaining if j != k]
    if len(cand) < 2:
        return 1.0
    srt = sorted(cand, reverse=True)
    return float(srt[0] - srt[1])

def _novel_rate_from_analyze(equal, parent_children, unassigned, new_adata, annotation):
    """
    基于 analyze_transmission 的 unassigned 结果，定义“新类比例”（Not in D1）
    分母：new_adata 中的 unique 注释数
    """
    total = max(1, new_adata.obs[annotation].nunique())
    novel_cnt = sum(1 for k, v in unassigned.items() if v == 'Not in D1')
    return float(novel_cnt) / total

# ===========================================================================================


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
    （保持你的原逻辑，不改）
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

    elif (not first_call) and (not failed_call):
        adata_new0 = adata[adata.obs[annotation].isin(parent_list)]
        unassigned_to_remove = []
        parent_list = []

    elif failed_call:
        unassigned_to_remove = []
        parent_list = find_parent_labels(tree, parent_list)
        adata_new0 = adata[adata.obs[annotation].isin(parent_list)]
        failed_assigned = []

    D2_list = []

    df_OT_T2S_new, df_OT_S2T_new = count_confumatrix(adata_new0, new_adata, annotation, transmission_ratio)
    equal_new, parent_children_new, unassigned_new = analyze_transmission(df_OT_S2T_new, df_OT_T2S_new)

    for unassigned_node in unassigned_new.keys():
        if unassigned_new[unassigned_node] == 'Not in D2':
            unassigned_to_remove.append(unassigned_node)

    for unassigned_node in unassigned_to_remove:
        if unassigned_node in unassigned_new:
            unassigned_new.pop(unassigned_node)

    for unassigned in unassigned_new:
        if (unassigned not in failed_assigned) and (unassigned_new[unassigned] == 'Not in D1'):
            failed_assigned.append(unassigned)

    equal_to_remove = []
    for cell in equal_new:
        if equal_new[cell] in failed_assigned:
            tree.find_node(cell).addkid(myNode(equal_new[cell]))
            failed_assigned.remove(equal_new[cell])
            equal_to_remove.append(cell)
    for cell in equal_to_remove:
        equal_new.pop(cell)

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

    if not failed_call:
        adata = merge_annotations(adata, annotation, equal_new)
        add_equal_anno(tree, equal_new)

    for parent, children in parent_children_new.items():
        child_preparent = []
        for child in children:
            if tree.find_node(child) is not None:
                child_preparent.append(tree.find_node(child).parent.label)

        for child in children:
            if (tree.find_node(child) is not None) and (tree.find_node(child).parent.label != "root"):
                if tree.find_node(child).parent.label not in parent_list:
                    parent_list.append(tree.find_node(child).parent.label)

                if parent not in D2_list:
                    D2_list.append(parent)
                chilren_record_dict[parent] = children

            elif tree.find_node(child) is None:
                childnode = myNode(child)
                tree.find_node(parent).addkid(childnode)
                if child in chilren_record_dict:
                    for i in chilren_record_dict[child]:
                        previous_parent = tree.find_node(i).parent
                        tree.find_node(i).parent = childnode
                        previous_parent.children.remove(tree.find_node(i))

            elif (tree.find_node(child).parent.label == "root") & (all(item == "root" for item in child_preparent)):
                if tree.find_node(parent) is None:
                    childnode = myNode(parent)
                    tree.addkid(childnode)
                else:
                    childnode = tree.find_node(parent)
                tree.find_node(child).parent = childnode
                tree.children.remove(tree.find_node(child))

            elif (tree.find_node(child).parent.label == "root") & (any(item != "root" for item in child_preparent)):
                parent_list.append(child)

    parent_list = [label for label in parent_list if (tree.find_node(label).parent.label == "root" or tree.find_node(label).parent.label not in parent_list)]

    if (D2_list != []) & (failed_call == False):
        new_adata1 = new_adata[new_adata.obs[annotation].isin(D2_list)]
        subsequent_comparisons(tree, new_adata1, adata, annotation, transmission_ratio, parent_list, chilren_record_dict, failed_assigned,
                               first_call=False, failed_call=False, iteration=True)

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

def do_harmonization(adata, annotation_index, batch_index, sample_size=100, transmission_ratio=0.4, batch_order=None, return_audit=False):
    """
    Perform automatic comparison and harmonization of all batches in the adata object.

    新增：顺序体检（指标 1/2/3）
      - 每一步记录 sim_to_pool, top2_gap, novel_rate, novel_delta
      - 可选返回审计表 audit_df 以及 pairwise 相似度矩阵 S
    """
    start_time = time.time()

    # --------- (A) 为顺序体检准备：用“未加前缀”的原始注释构建 pairwise S ----------
    # 先保留一份“原始注释”的副本，用于 pairwise 评估（不受批次前缀影响）
    _adata_raw = adata.copy()

    # 按 batch 划分原始数据（未加 batch- 前缀）
    batches_raw = _adata_raw.obs[batch_index].unique().tolist()
    batch_raw_dict = {}
    for b in batches_raw:
        batch_raw_dict[b] = _adata_raw[_adata_raw.obs[batch_index] == b]

    # 根据用户给定的顺序或注释规模排序，生成最终顺序的 batch 名称列表
    if batch_order is None:
        _counts = [(b, batch_raw_dict[b].obs[annotation_index].value_counts().sum()) for b in batches_raw]
        _counts.sort(key=lambda x: x[1], reverse=True)
        ordered_batch_names = [x[0] for x in _counts]
    else:
        if not isinstance(batch_order, list):
            raise ValueError("batch_order should be a list. Please provide a list of batch orders.")
        ordered_batch_names = sorted(batches_raw, key=lambda x: batch_order.index(x) if x in batch_order else len(batch_order))

    # 为体检构建 pairwise 相似度矩阵（用原始注释）
    # 注意：这里 S 的行列顺序与 ordered_batch_names 一致
    _batch_raw_ordered = {bn: batch_raw_dict[bn] for bn in ordered_batch_names}
    names_for_S, S = _build_pairwise_S(_batch_raw_ordered, annotation_index, transmission_ratio)
    name_to_idx = {n: i for i, n in enumerate(names_for_S)}

    # --------- (B) 下面进入你原本的对齐流程（加 batch- 前缀），并同时做“旁路记录” ----------
    root = myNode("root")

    # 在对齐流程里，你会把注释改为 "batch-anno"；这里保持你的原逻辑不变
    adata.obs[annotation_index] = adata.obs[batch_index].astype(str) + '-' + adata.obs[annotation_index].astype(str)

    adataori = adata.copy()
    adata = adata.copy()

    # 重新得到排序后的批次（与 ordered_batch_names 对齐）
    batches = ordered_batch_names
    batch_counts = [(b, (adata[adata.obs[batch_index] == b]).obs[annotation_index].value_counts().sum()) for b in batches]

    batch_data_dict = {}
    for i, (batch, _) in enumerate(batch_counts):
        batch_data_dict[f"adata{i}"] = adata[adata.obs[batch_index] == batch]

    batch_names = list(batch_data_dict.keys())

    # ====== 新增：顺序体检记录容器 ======
    audit_rows = []
    prev_novel_rate = None

    # 构造 “顺序中的索引映射”：外部序列中的第 i 步，对应 names_for_S 中的 idx
    order_idx = [name_to_idx[bname] for bname in ordered_batch_names]
    used_mask = {i: False for i in range(len(order_idx))}
    pool_mask = {i: False for i in range(len(order_idx))}

    for i in range(len(batch_names)):
        batch_name_readable = batch_counts[i][0]  # 实际的 batch 名称
        if i == 0:
            adata0 = batch_data_dict[batch_names[i]]
            pool_mask[order_idx[i]] = True  # 第一批进入“已加入池”

            # 体检：第一步（池为空）只能记录 top2_gap（以剩余者为参照）和 novel_rate（=0，因还未对齐）
            sim_pool = 0.0
            gap = _top2_gap(S, order_idx, i, used_mask)
            novel_rate = 0.0
            novel_delta = 0.0
            audit_rows.append({
                "step": i+1,
                "dataset": batch_name_readable,
                "sim_to_pool": sim_pool,
                "top2_gap": gap,
                "novel_rate": novel_rate,
                "novel_delta": novel_delta
            })
            used_mask[order_idx[i]] = True

        elif i == 1:
            adata1 = batch_data_dict[batch_names[i]]
            print(f"Initializing tree from {batch_counts[0][0]} and {batch_counts[1][0]}🌲")

            df_OT_T2S, df_OT_S2T = count_confumatrix(adata0, adata1, annotation_index, transmission_ratio)
            equal, Parent_kids, unassigned = analyze_transmission(df_OT_S2T, df_OT_T2S)

            # 体检：第二步
            sim_pool = _sim_to_pool(S, order_idx, i, pool_mask)  # 与“池（只有第一批）”的平均相似度
            gap = _top2_gap(S, order_idx, i, used_mask)
            # 基于初始化这一步的 unassigned 计算 novel_rate（Not in D1）
            novel_rate = _novel_rate_from_analyze(equal, Parent_kids, unassigned, adata1, annotation_index)
            novel_delta = (novel_rate - (prev_novel_rate if prev_novel_rate is not None else 0.0))
            audit_rows.append({
                "step": i+1,
                "dataset": batch_name_readable,
                "sim_to_pool": sim_pool,
                "top2_gap": gap,
                "novel_rate": novel_rate,
                "novel_delta": novel_delta
            })
            prev_novel_rate = novel_rate
            used_mask[order_idx[i]] = True
            pool_mask[order_idx[i]] = True

            adataori = merge_annotations(adataori, annotation_index, equal)
            root = initialize_tree(equal, Parent_kids, unassigned)

        else:
            adata_subs = batch_data_dict[batch_names[i]]
            print(f"Adding annotations from {batch_counts[i][0]}🍃")

            # 体检：加入前先算 sim_to_pool / top2_gap
            sim_pool = _sim_to_pool(S, order_idx, i, pool_mask)
            gap = _top2_gap(S, order_idx, i, used_mask)

            # —— 为了拿到“本步”的 novel_rate，我们需要一次与树的比较；
            # 你的 subsequent_comparisons 内部会多次调用 count_confumatrix，
            # 但其中首次（first_call=True）刚好会做一轮与根/顶层的比较，我们沿用“第二步的口径”：
            #   novel_rate = ‘Not in D1’ / new_adata 的 unique 注释数
            # 为了不改你现有函数，这里复用 count_confumatrix(当前树顶层子集, 新批) 的一次粗评：
            # 使用树顶层（root.kids）在 adataori 中的子集作为 D1，模拟“第一轮规则”。
            root_top_labels = root.kids
            adata_unassigned_like = adataori[adataori.obs[annotation_index].isin(root_top_labels)]
            df_T2S_tmp, df_S2T_tmp = count_confumatrix(adata_unassigned_like, adata_subs, annotation_index, transmission_ratio)
            equal_tmp, parent_kids_tmp, unassigned_tmp = analyze_transmission(df_S2T_tmp, df_T2S_tmp)
            novel_rate = _novel_rate_from_analyze(equal_tmp, parent_kids_tmp, unassigned_tmp, adata_subs, annotation_index)
            novel_delta = (novel_rate - (prev_novel_rate if prev_novel_rate is not None else 0.0))

            # 记录体检行（加入前）
            audit_rows.append({
                "step": i+1,
                "dataset": batch_name_readable,
                "sim_to_pool": sim_pool,
                "top2_gap": gap,
                "novel_rate": novel_rate,
                "novel_delta": novel_delta
            })
            prev_novel_rate = novel_rate
            used_mask[order_idx[i]] = True
            pool_mask[order_idx[i]] = True

            # —— 正式执行你的对齐（不改动你的逻辑）
            subsequent_comparisons(root, adata_subs, adataori, annotation_index, transmission_ratio)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Batch comparison completed. Time elapsed: {elapsed_time:.4f} seconds")

    if return_audit:
        audit_df = pd.DataFrame(audit_rows, columns=["step","dataset","sim_to_pool","top2_gap","novel_rate","novel_delta"])
        return root, audit_df, (names_for_S, S)
    return root
