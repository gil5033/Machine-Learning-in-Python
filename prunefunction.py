from sklearn.tree._tree import TREE_LEAF

def prune_function( sub_tree, index, threshold ):
 # If the count one of the classes is less than threshold, turn that node into a leaf
 # sub_tree.value[ index ] is the count of each class of outcome at that node

    if sub_tree.value[ index ].min() < threshold:
    # Turn the node into a leaf by unlinking its child nodes
    # TREE_LEAF is (effectively) NULL or null_ptr
    
        sub_tree.children_left[ index ] = TREE_LEAF
        sub_tree.children_right[ index ] = TREE_LEAF
    # Otherwise, visit the child nodes
    # if the children_left[] == NULL, then children_right[] == NULL
    if sub_tree.children_left[ index ] != TREE_LEAF:

    # This is where the recursion happens

        prune_function( sub_tree, sub_tree.children_left[ index ], threshold )
        prune_function( sub_tree, sub_tree.children_right[ index ], threshold )