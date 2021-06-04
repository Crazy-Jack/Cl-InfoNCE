import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import time
import os


def entropy(x):
    '''
    H(x)
    '''
    unique, count = np.unique(x, return_counts=True, axis=0)
    prob = count/len(x)
    H = np.sum((-1)*prob*np.log2(prob))

    return H

def joint_entropy(x, y):
    '''
    H(x,y)
    '''
    combine = np.c_[x, y]
    return entropy(combine)

def conditional_entropy(x, y):
    '''
    H(x|y)
    '''
    return joint_entropy(x, y) - entropy(y)

def mutual_information(x, y):
    '''
    I(x;y)
    '''
    return entropy(x) - conditional_entropy(x, y)

def get_MI_and_H(df, gran_lvl):
    '''
    input:
        - df: pandas dataframe
        - gran_lvl: latent class granularity level

    output:
        - MI: mutual information of Y and T
        - H: conditional entropy of Y given T
    '''

    T = df['class'].tolist()
    Y = df[gran_lvl].tolist()

    H = conditional_entropy(Y, T)
    MI = mutual_information(Y, T)

    return MI, H

# naively build hierchical clustering
def get_hirec_class_labels(instance_num, branch_num=2):
    """build hirecal latent class labels for certain class. 
    Return a dictionary: gran_lvl 0 represents instance id; max is all zero represent they are all the same class
    Param: 
      - instance_num: how many instance in this class
      - branch_num: group how many latent class from last layer to build next layer latent class
    
    Example:
    >>> instance_num = 16
    >>> get_hirec_class_labels(instance_num, branch_num=2)
    """
    index = np.arange(instance_num, dtype=np.int32)
    label = index
    gran_lvl = 0
    label_dict = {gran_lvl: label}
    while branch_num ** (gran_lvl + 1) < instance_num:
        gran_lvl += 1
        label = np.repeat(label, branch_num)
        label_dict[gran_lvl] = label

    label_dict[gran_lvl + 1] = np.zeros(instance_num, dtype=np.int32)

    label_for_class = {i: label_dict[i][:instance_num] for i in label_dict}
    
    return label_for_class

def get_gran_lvl(intend_lvl, build_latent_class):
    """Get gran_lvl for each class given specific requied granularity. if intended ganularity is larger than the internal granlarity, take the max
    """
    max_gran_class = [len(build_latent_class[i]) for i in build_latent_class]
    class_gran_lvl = [min(intend_lvl, i - 1) for i in max_gran_class]
    return class_gran_lvl

def get_new_latent_class(gran_lvl, build_latent_class):
    """
    Get new latent class: each class has a hirechical structure, higher gran_lvl contain lower gran_lvl
    """
    gran_info = get_gran_lvl(gran_lvl, build_latent_class)

    class_labels = {i:build_latent_class[i][gran_info[i]] for i in build_latent_class} # a dict: key is the class number, value is the array corresponds to each label inside the class

    new_class = {}
    class_count = 0
    for i in class_labels:
        this_class_label = class_labels[i]
        label_now = this_class_label + class_count
        new_class[i] = label_now
        class_count += np.unique(this_class_label).shape[0]

    return new_class

def build_all_labels(gran_lvl, class_stats, branch_num=2):
    """Take in gran_lvl, return {class_number: [class_instance_num,]}, each instance belongs to a latent class that is built for that gran_lvl
    gran_lvl: 0 represents instance id
              if pass a huge number, it will just return class num
    """
    # build entire latent class
    # {class_number: {gran_lvl: label within each class for this gran_lvl}}
    build_latent_class = {i: get_hirec_class_labels(class_stats[i], branch_num=branch_num) for i in class_stats}

    return get_new_latent_class(gran_lvl, build_latent_class)


def get_new_class_column(class_latent_labels, class_info):
    """convert class_latent_labels dict into a array of latent class labels"""
    class_info_np = np.array(class_info)
    new_latent_class = np.zeros(class_info_np.shape, dtype=np.int32)
    for i in class_latent_labels: # i is class label
        class_index = np.where(class_info_np == i)[0]
        np.random.seed(0) # same permutation for the same class for different gran_lvl
        class_index = np.random.permutation(class_index)
        new_latent_class[class_index] = class_latent_labels[i]
    return new_latent_class

def get_latent_all_pd(class_stats, class_info, branch_num=2):
    """input gran_lvl, class_stats, output the reassigned class assignment for that gran_lvl"""

    # assign class in table, every column is for one gran_lvl
    new_latent_class_dict = {}
    max_gran_lvl = max([max(get_hirec_class_labels(class_stats[i])) for i in class_stats])
    gran_lvl_list = np.arange(max_gran_lvl + 1) # control 
    for gran_lvl in gran_lvl_list:
        start_time = time.time()
        class_latent_labels = build_all_labels(gran_lvl, class_stats, branch_num=branch_num)
        gran_lvl_name = 'label_gran_{}'.format(max_gran_lvl - gran_lvl) # !!!! caveat: in order to comply with original setting(SupCon is gran_lvl 0 and the more closer to simCLR, the bigger the gran_lvl is), the column name is changed
        new_latent_class_dict[gran_lvl_name] = get_new_class_column(class_latent_labels, class_info)
        end_time = time.time()
        print("Done for gran_lvl {}; time: {} s".format(gran_lvl, end_time - start_time))
    new_latent_class_pd = pd.DataFrame(new_latent_class_dict)
    return new_latent_class_pd, 'label_gran_{}'.format(max_gran_lvl)


def check_same_class(final_hierc_latent, class_info_np):
    class_q = final_hierc_latent['label_gran_0']
    return np.all(np.array(class_q) == class_info_np)

######################
#    Load data       #
######################
data_meta_train_path = "../rank_H/meta_data_train.csv"
data_meta_test_path = "../rank_H/meta_data_test.csv"
PATH = "./"
meta_pd_train = pd.read_csv(data_meta_train_path, index_col=0)

class_num = meta_pd_train.iloc[-1]['class']
class_info = meta_pd_train.drop([-1])['class']
class_info_np = np.array(class_info)

class_stats = {}
for i in range(class_num):
    class_stats[i] = np.where(class_info_np == i)[0].shape[0]

class_stats_list = [class_stats[i] for i in class_stats]
class_stats_np = np.array(class_stats_list)
class_stats_np / class_stats_np.sum()


######################
#  (1): Fix H(Z|T)   #
######################
unique_class_info = np.unique(class_info_np)
unique_class_info

np.random.seed(0)
rand_unique_class = np.random.permutation(unique_class_info)
rand_unique_class

t_hirec_dict = get_hirec_class_labels(len(unique_class_info), branch_num=2)
new_latent_class_fix_h = {}
for m in t_hirec_dict:
    gran_lvl_class_map = t_hirec_dict[m]
    class_mapping = {rand_unique_class[i]:gran_lvl_class_map[i] for i in range(gran_lvl_class_map.shape[0])}
    new_latent_class = np.array([class_mapping[i] for i in class_info_np])
    gran_name = "label_gran_{}".format(m)
    new_latent_class_fix_h[gran_name] = new_latent_class

new_latent_class_fix_h_pd = pd.DataFrame(new_latent_class_fix_h)
new_latent_class_fix_h_pd['class'] = class_info_np
new_latent_class_fix_h_pd['path'] = list(meta_pd_train.drop([-1])['path'])

calculate_total_num = new_latent_class_fix_h_pd.apply(lambda x: len(set(x)), axis=0)
new_latent_class_fix_h_pd.loc[-1] = calculate_total_num

path = PATH
os.makedirs(path, exist_ok=True)
new_latent_class_fix_h_pd_path = os.path.join(path, 'meta_data_train_bran_2_hirc_rand_fixh.csv')
new_latent_class_fix_h_pd.to_csv(new_latent_class_fix_h_pd_path)


######################
#  (2): Fix I(Z;T)   #
######################
branch_num = 2
final_hierc_latent, max_gran = get_latent_all_pd(class_stats, class_info, branch_num=branch_num)
check_same_class(final_hierc_latent, class_info_np)

final_hierc_latent['class'] = final_hierc_latent['label_gran_0']
final_hierc_latent['path'] = list(meta_pd_train.drop([-1])['path'])

calculate_total_num = final_hierc_latent.apply(lambda x: len(set(x)), axis=0)
final_hierc_latent.loc[-1] = calculate_total_num

path = PATH
os.makedirs(path, exist_ok=True)
train_new_latent_class_path = os.path.join(path, 'meta_data_train_bran_2_hirc_rand.csv')
final_hierc_latent.to_csv(train_new_latent_class_path)

######################
#  (3): Fix H(Z)     #
######################

def permutate_touched_latent_class(untouched_classes, class_info_np, gran_lvl_info):
    """untouch certain class num latent class, permute the rest (reserve H(Y))"""
    # get untouched instance index
    untouched_instance_index = []
    for i in untouched_classes:
        index = np.where(class_info_np == i)[0]
        untouched_instance_index.append(index)
    untouched_instance_index_np = np.concatenate(untouched_instance_index)

    
    # permutate touched id
    my_gran_lvl_info = gran_lvl_info * np.ones(gran_lvl_info.shape) # replicate the gran_lvl_info
    untouched_latent_class_np = my_gran_lvl_info[untouched_instance_index_np]
    touched_index = np.delete(np.arange(my_gran_lvl_info.shape[0]), untouched_instance_index_np, 0) # exclude untouched index
    tourched_latent_class = my_gran_lvl_info[touched_index]
    my_gran_lvl_info[touched_index] = np.random.permutation(tourched_latent_class)
    
    return my_gran_lvl_info.astype(np.int32)


def get_perm_column(class_stats, class_permu, class_info_np, gran_lvl_info, gran_lvl):
    """get conditional entropy and mutual information for given granularity"""
    entropy_y_g_t_s = []
    mi_y_t_s = []
    permutated_table = {}

    for i in class_stats:
        touched_classes = class_permu[:i]
        untouched_classes = [i for i in class_stats if i not in touched_classes]
        permutated_column = permutate_touched_latent_class(untouched_classes, class_info_np, gran_lvl_info)
        permutated_table['label_gran_{}_{}'.format(gran_lvl, i)] = permutated_column
        
        print('generating for label_gran_{}_{}'.format(gran_lvl, i))
        entropy_y_g_t = conditional_entropy(permutated_column, class_info_np)
        mi_y_t = mutual_information(permutated_column, class_info_np)
        entropy_y_g_t_s.append(entropy_y_g_t)
        mi_y_t_s.append(mi_y_t)
    
    return entropy_y_g_t_s, mi_y_t_s, permutated_table


class_permu = np.random.permutation(list(class_stats.keys()))
# entropy_y_g_t_s, mi_y_t_s, permutated_table = get_perm_column(class_stats, class_permu, class_info_np,  gran_lvl_info, 0)
permutated_table_s = {}
entropy_y_g_t_ss = {}
mi_y_t_ss = {}
final_hierc_latent = final_hierc_latent.drop([-1])

for i in final_hierc_latent:
    if i not in ['class', 'path']:
        gran_lvl = int(i.split("_")[-1])
        gran_lvl_info = np.array(final_hierc_latent[i])
        entropy_y_g_t_s, mi_y_t_s, permutated_table = get_perm_column(class_stats, class_permu, class_info_np,  gran_lvl_info, gran_lvl)
        permutated_table_s[gran_lvl] = permutated_table
        entropy_y_g_t_ss[gran_lvl] = entropy_y_g_t_s
        mi_y_t_ss[gran_lvl] = mi_y_t_s

permutated_table_all = {k: v for d in permutated_table_s for k, v in permutated_table_s[d].items()}
permutated_table_all_pd = pd.DataFrame(permutated_table_all)
permutated_table_all_pd['class'] = class_info_np
permutated_table_all_pd['path'] = list(meta_pd_train.drop([-1])['path'])

calculate_total_num = permutated_table_all_pd.apply(lambda x: len(set(x)), axis=0)
permutated_table_all_pd.loc[-1] = calculate_total_num

path = PATH
os.makedirs(path, exist_ok=True)
permutated_table_all_pd_path = os.path.join(path, 'meta_data_train_bran_2_fix_allh_from_mi.csv')
permutated_table_all_pd.to_csv(permutated_table_all_pd_path)