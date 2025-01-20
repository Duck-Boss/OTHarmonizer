from zss import Node
import zss
import itertools

########### new tree-node class for PCBS and AHA ###########
class myNode:
    def __init__(self, label):
        self.label = label
        self._parent = None
        self.children = []

    def addkid(self, node):
        if node not in self.children:
            node.parent = self
        return self

    def children(self):
        return self.children

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

    def __repr__(self, level=0):
        ret = "----" * level + repr(self.label) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret
############################################################

##################### utils functions ######################  
def convert_tree(zss_node):
    # Create a new node with the same label as the zss_node
    label = zss.Node.get_label(zss_node)
    new_node = myNode(label)

    # Recursively add children
    for child in zss.Node.get_children(zss_node):
        new_child = convert_tree(child)  # Convert the child
        new_node.addkid(new_child)  # Add the converted child to the new node

    return new_node

def process_node_label(node):
    node.label = {' & '.join(sorted(node.label))}
    for child in node.children:
        process_node_label(child)

def update_label(node):
    for child in node.children:
        update_label(child)
    node.label = set(node.label).union(*(child.label for child in node.children))

def count_diffs(tree1, tree2):
    def collect_labels(node):
        labels = [node.label]
        for child in node.children:
            labels.extend(collect_labels(child))
        return labels

    list1 = collect_labels(tree1)
    list2 = collect_labels(tree2)
    total_diffs = [i for i in list1 if i not in list2] + [i for i in list2 if i not in list1]
    return len(total_diffs), len(list1) + len(list2)

def collect_sets(node):
    sets = []
    labels = node.label
    if node.children:  
        for child in node.children:
            labels = labels.union(child.label)
            sets.extend(collect_sets(child))  
        sets.append(labels)
        
    return sets

def count_diff_lists(list1, list2):
    total_diffs = [i for i in list1 if i not in list2] + [i for i in list2 if i not in list1]
    return len(total_diffs), len(list1) + len(list2)


def compute_match_score(list1, list2):
    score = 0
    for set1 in list1:
        max_overlap = 0
        for set2 in list2:
            overlap = len(set1 & set2)
            if overlap > max_overlap:
                max_overlap = overlap
        score = score + max_overlap / len(set1)
    for set2 in list2:
        max_overlap = 0
        for set1 in list1:
            overlap = len(set1 & set2)
            if overlap > max_overlap:
                max_overlap = overlap
        score = score + max_overlap / len(set2)
    return score

def get_parent_child_pairs(node):
    result = []
    for child in node.children:
        result.append(node.label.union(child.label))
        result.extend(get_parent_child_pairs(child))
    return result

def find_path(node):
    path = []
    while node is not None:
        path.append(node.label)
        node = node.parent
    return path[::-1]  # return path from root to the input node

def determine_relationship(path1, path2):
    len1, len2 = len(path1), len(path2)
    min_len = min(len1, len2)
    if min_len == 0:
        return "X"
    for i in range(min_len):
        if path1[i] != path2[i]:
            return "∧"
    if len1 > len2:
        return "⊆"
    elif len1 < len2:
        return "⊇"
    else: 
        return "="

def traverse_and_map(node, sample_to_node):
    if hasattr(node, 'label'):
        for sample_id in node.label:
            sample_to_node[sample_id] = node
    for child in node.children:
        traverse_and_map(child, sample_to_node)
        
def tree_to_relationships(tree, Anno_list):
    # construct dict from Annotation to tree
    Anno_to_node = {}
    traverse_and_map(tree, Anno_to_node)
    # translate 
    relationships = []
    for batch1_Anno, batch2_Anno in itertools.combinations(Anno_list, 2):
        for Anno1 in batch1_Anno:
            for Anno2 in batch2_Anno:
                if Anno1 != Anno2:
                    path1 = find_path(Anno_to_node.get(Anno1))
                    path2 = find_path(Anno_to_node.get(Anno2))
                    relation = determine_relationship(path1, path2)
                    relationships.append((Anno1, Anno2, relation))
    return relationships

def categorize_by_batch(all_annos):
    categorized = {} 
    for s in all_annos:
        key = s.split('-')[0]
        if key not in categorized:
            categorized[key] = []
        categorized[key].append(s)
    return categorized

def find_common_elements(lst1, lst2):
    common_elements = [item for item in lst1 if item in lst2]
    return common_elements


def capture_relationships(node, relationships=None, seen_equal=None):
    if relationships is None:
        relationships = {
            "parent-child": [],
            "equal": [],
            "sum": [],
            "none-relation": []
        }
    if seen_equal is None:
        seen_equal = set()  # Track the equal relationships to avoid duplicates

    if '&' in node.label:
        parents = node.label.split('&')
    else:
        parents = [node.label]

    # Capture parent-child relationships
    for child in node.children:
        if '&' in child.label:
            parts = child.label.split('&')
            for parent in parents:
                for label in parts:
                    relationships["parent-child"].append(f"{parent}:{label}")
        else:
            for parent in parents:
                relationships["parent-child"].append(f"{parent}:{child.label}")
        capture_relationships(child, relationships, seen_equal)

    # Capture equal relationships (based on '&' symbol)
    all_labels = node.get_all_descendants_labels()

    for label in all_labels:
        if '&' in label:
            parts = label.split('&')
            # Generate equal relationships for each unique pair of parts
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    # Create a sorted tuple to ensure we don't repeat pairs
                    pair = tuple(sorted([parts[i], parts[j]]))
                    if pair not in seen_equal:
                        seen_equal.add(pair)
                        relationships["equal"].append({parts[i], parts[j]})

    # Combine "equal" and "parent-child" into "sum"
    relationships["sum"] = relationships["equal"] + relationships["parent-child"]

    # Calculate none-relation
    # All labels in the tree
    all_nodes = set()
    for label in all_labels:
        if '&' in label:
            all_nodes.update(label.split('&'))
        else:
            all_nodes.add(label)

    # Check for unlinked relationships
    for label1 in all_nodes:
        for label2 in all_nodes:
            if label1 != label2:
                # Skip if the pair is in parent-child or equal or sum
                pair1 = f"{label1}:{label2}"
                pair2 = f"{label2}:{label1}"
                equal_pair = {label1, label2}
                if pair1 not in relationships["parent-child"] and \
                   pair2 not in relationships["parent-child"] and \
                   equal_pair not in relationships["equal"] :
                    # Add to none-relation
                    none_relation = set(sorted([label1, label2]))
                    if none_relation not in relationships["none-relation"]:
                        relationships["none-relation"].append(none_relation)

    return relationships

def count_f1(querylist, reflist):
    common = find_common_elements(querylist, reflist)
    recall = len(common)/len(reflist)
    precision = len(common)/len(querylist)
    if recall == 0 and precision == 0:
        f1 = 0
    else:
        f1 = 2*recall*precision/(recall + precision)
    
    
    return f1


############################################################


##################### metric functions #####################
def TEDS(constructed_tree, ref_tree, root_node_name= "root"):
    ### Tree Edit Distance Similarity###
    root_node = Node({root_node_name})
    dist_constructed_tree = zss.simple_distance(constructed_tree, root_node)
    print("nodes of constructed_tree", dist_constructed_tree)
    dist_ref_tree = zss.simple_distance(ref_tree, root_node)
    print("nodes of ref_tree", dist_ref_tree)
    
    dist = zss.simple_distance(constructed_tree, ref_tree)
    print("TED:", dist)
    #TEDS = 1 - dist / max(dist_constructed_tree, dist_ref_tree)
    TEDMS = 1 - dist / (dist_constructed_tree + dist_ref_tree - 1)
    #print("TEDS:", TEDS)
    print("TEDS:", TEDMS)
    return dist, TEDMS
    
def PCBS(constructed_tree, ref_tree):
    ### convret tree to the seconde form ###
    new_constructed_tree = convert_tree(constructed_tree)
    new_ref_tree = convert_tree(ref_tree)
    
    process_node_label(new_constructed_tree)
    process_node_label(new_ref_tree)
    
    ### Parent-Children Branch Similarity ###
    constructed_tree_lists = collect_sets(new_constructed_tree)
    ref_tree_lists = collect_sets(new_ref_tree)
    
    RF_dist_2level, total_counts_2level = count_diff_lists(constructed_tree_lists, ref_tree_lists)
    print("total count of sets", total_counts_2level)
    print("PCB:", RF_dist_2level)
    RFS = 1 - RF_dist_2level / total_counts_2level
    
    RF_simi_2level = compute_match_score(constructed_tree_lists, ref_tree_lists)
    print("PCBS:", RF_simi_2level / (len(constructed_tree_lists) + len(ref_tree_lists)))
                                                           
    return RF_dist_2level, RFS, RF_simi_2level / (len(constructed_tree_lists) + len(ref_tree_lists))

def AH_F1(querytree, reftree):
    relationships1 = capture_relationships(querytree)['sum']
    relationships2 = capture_relationships(reftree)['sum']
    relationships1_parent = capture_relationships(querytree)['parent-child']
    relationships2_parent = capture_relationships(reftree)['parent-child']
    relationships1_equal = capture_relationships(querytree)['equal']
    relationships2_equal = capture_relationships(reftree)['equal']
    relationships1_non = capture_relationships(querytree)['none-relation']
    relationships2_non = capture_relationships(reftree)['none-relation']
    
    parentf1 = count_f1(relationships1_parent, relationships2_parent)
    equalf1 = count_f1(relationships1_equal, relationships2_equal)
    nonef1 = count_f1(relationships1_non, relationships2_non)
    
    AH_F1 = (parentf1+equalf1+nonef1)/3
    print("AH_F1:", AH_F1)
    
    return AH_F1
############################################################

def benchmark(constructed_tree, ref_tree, metrics=["TEDS", "PCBS", "AH_F1"], print_trees=False):
    metrics_results = {}
    if "TEDS" in metrics:
        _TED, _TEDS = TEDS(constructed_tree, ref_tree)
        metrics_results["TEDS"] = _TEDS
        print("###################################")
    if  "PCBS" in metrics:
        _PCB, _PCBS, _PCB_P = PCBS(constructed_tree, ref_tree)
        metrics_results["PCBS"] = _PCB_P
        print("###################################")
    if "AH_F1" in metrics:
        _AH_F1 = AH_F1(constructed_tree, ref_tree)    
        metrics_results["AH_F1"] = _AH_F1
        print("###################################")
    ### print two trees ###
    if print_trees:
        print('\n\n')
        print("ref_tree:")
        print('\n')
        print(new_ref_tree)
        print('\n\n')
        print("constructed_tree:")
        print('\n')
        print(new_constructed_tree)
    return metrics_results