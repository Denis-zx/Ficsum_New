from skmultiflow.data.random_tree_generator import RandomTreeGenerator


stream = RandomTreeGenerator(tree_random_state=8873, sample_random_seed=69, n_classes=2,
n_cat_features=2, n_num_features=5, n_categories_per_cat_feature=5, max_tree_depth=6,
 min_leaf_depth=3, fraction_leaves_per_level=0.15)