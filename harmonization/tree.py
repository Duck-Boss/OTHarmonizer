from .harmonize import zssNode
from .harmonize import myNode



def clean_string(input_str):
    cleaned = input_str.lstrip('-').strip()
    cleaned = cleaned.strip("'")
    return cleaned

def convert_to_zssNode(my_node):
    label_set = set(my_node.label.split('&'))
    zss_node = zssNode(label_set)
    for child in my_node.children:
        zss_node.addkid_with_order(convert_to_zssNode(child))
    return zss_node

def convert_to_myNode(zss_node):
    label_str = '&'.join(zss_node.label)
    my_node = myNode(label_str)
    for child in zss_node.children:
        my_node.addkid(convert_to_myNode(child))
    return my_node


def create_tree_from_string(tree_string):
    lines = tree_string.strip().split('\n')
    root = None
    nodes = []
    current_indent = -1

    for line in lines:
        clean_line = line.strip().strip('"')
        indent_level = (len(line) - len(line.lstrip('----'))) // 4

        new_node = myNode(clean_string(clean_line))

        if indent_level == 0:
            root = new_node
            nodes = [root]
        else:
            parent_node = nodes[indent_level - 1]
            parent_node.addkid(new_node)

            if len(nodes) > indent_level:
                nodes[indent_level] = new_node
            else:
                nodes.append(new_node)

    return root

def create_tree_from_string(tree_string):
    lines = tree_string.strip().split('\n')
    root = None
    nodes = []
    current_indent = -1

    for line in lines:
        clean_line = line.strip().strip('"')
        indent_level = (len(line) - len(line.lstrip('----'))) // 4

        new_node = myNode(clean_string(clean_line))

        if indent_level == 0:
            root = new_node
            nodes = [root]
        else:
            parent_node = nodes[indent_level - 1]
            parent_node.addkid(new_node)

            if len(nodes) > indent_level:
                nodes[indent_level] = new_node
            else:
                nodes.append(new_node)

    return root