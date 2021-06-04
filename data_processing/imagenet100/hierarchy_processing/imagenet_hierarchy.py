"""hierarchy for testing"""


import os
import json
import argparse
from collections import defaultdict


import pandas as pd
import numpy as np
from tqdm import tqdm

from stats import conditional_entropy

def set_args():
    parser = argparse.ArgumentParser(
        "For interpolating hierachical data from json files")
    parser.add_argument("--root", default="/path/to/imagenet", help="root folder for imagenet images, \
                                                contains multiple train and val dataset, and different folders for label files")
    parser.add_argument("--json_file", default="wordnet_hierarchy_name.json",
                        help="hierarchy instruction based on this json file")
    parser.add_argument("--parent_child_file", default="wordnet.is_a.txt",
                        help="parent child relationship pair for wordnet obtained from ImageNet API page (is-a relationship)")
    parser.add_argument("--imagenet_class",
                        default="imagenet.synset.obtain_synset_list")
    parser.add_argument("--target_level", type=int, default=10, help="target level to defold")
    parser.add_argument("--interpolation_file", default="label_intepolate_pd.csv",
                        help="output interpolation instruction in csv format")
    parser.add_argument("--class_stats_file", type=str, default="class_stats.json", help="contain class statistics files")
    parser.add_argument("--class_path_pd_file", type=str, default="class_path.csv", help="path filename")
    parser.add_argument("--meta_file_train_target100", type=str, default="meta_data_train.csv", help="outputfilename for train")
    parser.add_argument("--meta_file_train_target100_source", type=str, default="meta_file_train_target100_source.csv", help="outputfilename for train")

    args = parser.parse_args()
    return args


############################
#           I/O            #
############################

def load_hierarchy(args):
    """load hiearchy json file in a safe manner"""
    print("Loading hierarchy json file...")
    with open(os.path.join(args.root, args.json_file), 'r') as f:
        hierarchy = json.load(f)
    return hierarchy


def load_pc_pairs(args):
    """load parent child pairs"""
    print("Load parent child pairs...")
    parents = defaultdict(list)
    with open(os.path.join(args.root, args.parent_child_file)) as f:
        for i in f.readlines():
            parent, child = i.replace("\n", "").split(" ")
            parents[child].append(parent)

    return parents


def load_imgnet_labels(args):
    """load id that is used by imagenet"""
    img_ids = []
    with open(os.path.join(args.root, args.imagenet_class)) as f:
        for i in f.readlines():
            img_ids.append(i.split(" ")[0])

    print(len(img_ids))

    return img_ids

def save_level_dict(level_dict, args, level):
    with open(os.path.join(args.root, f"level_{level}.json"), 'w') as f:
        json.dump(level_dict, f, sort_keys=True, indent=4)


def load_class_stats(args):
    """find out imagenet class statistics, i.e. how many instance is aviable for each class"""
    with open(os.path.join(args.root, args.class_stats_file), 'r') as f:
        class_stats = json.load(f)
    return class_stats
    
def load_image_path_class_pd(args):
    """load original image and path class pandas files"""
    class_pd = pd.read_csv(os.path.join(args.root, args.class_path_pd_file), index_col=0)
    return class_pd


############################
#           Build          #
############################
def find_max_level(hierarchy, max_level):
    """fine the max level inside the hierarchy"""
    for item in hierarchy:
        if item['Level'] > max_level:
            max_level = item['Level']

        if "Subcategory" in item.keys():
            max_level = max(max_level, find_max_level(
                item['Subcategory'], max_level))

    return max_level


def flat_hier(nodes, imgnet_labels):
    # iterative version
    queue = [(i, i['Meaning']) for i in nodes]
    out_nodes = []
    count = 0
    while queue != []:
        node, trace = queue.pop(0)
        if 'Subcategory' in node.keys():
            for child in node['Subcategory']:
                new_trace = trace + " -> " + child['Meaning']
                queue.append((child, new_trace))
                
                if child['LabelName'] in imgnet_labels:
                    one_node = {
                        'LabelName': child['LabelName'], 'Level': child['Level'], 'Meaning': child['Meaning'], 'InImageNet': child['InImageNet'], "LevelTrace": new_trace}
                    out_nodes.append(one_node)

                    count += 1

    return out_nodes, count


def build_level_hierarchy(hierarchy, imgnet_labels, target_level):
    queue = [(i, "start") for i in hierarchy]

    out_cluster = []

    while queue != []:
        node, trace = queue.pop(0)
        new_trace = trace + " -> " + node['Meaning']

        if node['Level'] < target_level:
            if 'Subcategory' in node.keys():
                if node['LabelName'] in imgnet_labels:

                    out_cluster.append(
                        {
                            'LabelName': node['LabelName'], 'Level': node['Level'], 'Meaning': node['Meaning'], 
                            'LevelTrace': new_trace,
                            "InImageNet": True if 'InImageNet' in node.keys() else False,
                        }
                    )
                for child in node['Subcategory']:

                    queue.append((child, new_trace))
            else:
                # if there is iamgenet target
                if node['LabelName'] in imgnet_labels:

                    out_cluster.append(
                        {
                            'LabelName': node['LabelName'], 'Level': node['Level'], 'Meaning': node['Meaning'], 
                            'LevelTrace': new_trace,
                            "InImageNet": True if 'InImageNet' in node.keys() else False,
                        }
                    )

        else:


            if 'Subcategory' in node.keys():

                flat_node, count = flat_hier([node], imgnet_labels)
                if count > 0:

                    out_cluster.append(
                        {
                            'LabelName': node['LabelName'], 'Level': node['Level'], 'Meaning': node['Meaning'], 
                            'Subcategory': flat_node,
                            'LevelTrace': new_trace,
                            "InImageNet": True if 'InImageNet' in node.keys() else False,
                        }
                    )
                elif node['LabelName'] in imgnet_labels:
                    out_cluster.append(
                        {
                            'LabelName': node['LabelName'], 'Level': node['Level'], 'Meaning': node['Meaning'], 
                            'LevelTrace': new_trace,
                            "InImageNet": node['InImageNet'],
                        }
                    )
            elif node['LabelName'] in imgnet_labels:

                    out_cluster.append(
                        {
                            'LabelName': node['LabelName'], 'Level': node['Level'], 'Meaning': node['Meaning'], 
                            'LevelTrace': new_trace,
                            "InImageNet": True if 'InImageNet' in node.keys() else False,
                        }
                    )



    return out_cluster



def pruned_imgnet_is_a(level_dict, imgnet_labels):
    """return a list of parent-child relationship"""
    queue = level_dict
    out_relationship = defaultdict(list)

    while queue != []:
        node = queue.pop(0)
        if "Subcategory" in node.keys():
            for child in node['Subcategory']:
                queue.append(child)
                if child['LabelName'] in imgnet_labels:
                    out_relationship[child['LabelName']].append(node['LabelName'])

    out_relationship = {i:list(set(out_relationship[i])) for i in out_relationship}
    return out_relationship



def select_subset_candidate(hierarchy, imgnet_labels, args):
    """
    filter out the class that could belong to multiple group across all level;
    return: classes that consistantly belongs to one group for every level
    """
    exclude_list = []
    out_relationship_all_level = {}
    for level in range(1, 18):
        # flatten certain level
        level_dict = build_level_hierarchy(hierarchy, imgnet_labels, level)

        # select unique parents
        out_relationship = pruned_imgnet_is_a(level_dict, imgnet_labels)
        out_relationship_all_level["level_" + str(level)] = out_relationship # {level_1: {<class_label>: [parents]}}
        # about multiple parents
        multiple_parents = {i:out_relationship[i] for i in out_relationship if len(out_relationship[i]) >= 2}

        print("multiple parents", multiple_parents)
        print("multiple parents", len(multiple_parents))
        print("outrelationship", len(out_relationship))
        exclude_list.extend([i for i in multiple_parents.keys()])
        print(f"Done for level {level}")

    # build cluster for this level
    exclude_list = list(set(exclude_list))
    usable_classes = [i for i in imgnet_labels if i not in exclude_list]
    print(f"exclude classes {exclude_list}")

    print(f"usable_classes {len(usable_classes)}")
    return usable_classes, out_relationship_all_level


def get_class_mapping(str_latent_class):
    '''
    assign numeric id to each unique attribute-combination
    '''
    class_appeared = set(str_latent_class)
    count = 0
    latent_class_mapping = {}
    for i in class_appeared:
        latent_class_mapping[i] = count
        count += 1
    return latent_class_mapping


def digitalize(level_assignment):
    """
    digitalize the class assignment from string to int
    """
    # digitalization
    mapping = get_class_mapping(level_assignment)
    level_assignment_np = np.array([mapping[i] for i in level_assignment])
    return level_assignment_np

def map_to_parent(t_class, parents_info_level):
    """
    parents_info_level: {<classid>: [parent]}, only contains classid that has a super-relationship 
    """
    if t_class in parents_info_level:
        assert len(parents_info_level[t_class]) < 2, f"{t_class} has two or more parents {parents_info_level[t_class]}"
        parent = parents_info_level[t_class][0]
    else:
        parent = t_class 
    return parent  

def find_target_class(target_classes, class_stats, out_relationship_all_level):
    """
    take in selected classes and build superclasses for different level
    param: 
        - class_stats: {'class_id': <num_of_instance> (-> int)}
        - target_class : [<class_id>]  
        - out_relationship_all_level: # {level_1: {<class_label>: [parents]}}
    """
    # baseline T
    class_info = []
    for class_id in target_classes:
        class_info.extend(class_stats[class_id] * [class_id])
    class_info_np = digitalize(class_info)

    # find out corresponding class for every level
    statistics = []
    for level in out_relationship_all_level:
        parents_info = out_relationship_all_level[level]
        
        level_assignment = []
        for t_class in target_classes:
            parent = map_to_parent(t_class, parents_info)
            level_assignment.extend([parent] * class_stats[t_class])
        
        # digitalization
        level_assignment_np = digitalize(level_assignment)

        # compute H(T|Y)
        statistics.append(conditional_entropy(class_info_np, level_assignment_np))

    print("Conditional Entropy: ", [round(i, 4) for i in statistics])
    # consecutive difference
    con_diff = [statistics[i+1]-statistics[i] for i in range(len(statistics)-1)]
    print("std of difference: ", np.std(con_diff))
    return statistics

def get_target_class_pd(class_path_pd, target_classes):
    print("Get target class pd from original...")
    target_pds = []
    for t_class in target_classes:
        target_pds.append(class_path_pd[class_path_pd['class'] == t_class])
    target_class_pd = pd.concat(target_pds)

    print("target class pd unqiue classes", len(set(target_class_pd['class'])))

    return target_class_pd


def build_class_meta_data_source(class_path_pd, target_classes, out_relationship_all_level):
    """
    given target classes, build dictionary for level lookup
    """
    target_class_pd = get_target_class_pd(class_path_pd, target_classes)
    print("Building meta data source...")
    
    for level in out_relationship_all_level:
        parents_info = out_relationship_all_level[level]
        level_parent = []
        for index in tqdm(range(len(target_class_pd)), total=len(target_class_pd)):
            t_class = target_class_pd.iloc[index, :]['class']
            if t_class in target_classes:
                parent = map_to_parent(t_class, parents_info)
                level_parent.append(parent)

        
        target_class_pd["label_gran_"+level] = level_parent
    
    return target_class_pd

def build_digitalized_meta_data(target_meta_file_train_source_pd):
    print("Digitalizing the source file...")
    target_meta_file_train_pd = {}
    for column in target_meta_file_train_source_pd.columns:
        if column in  ['path']:
            target_meta_file_train_pd[column] = target_meta_file_train_source_pd.loc[:, column]
        else:
            target_meta_file_train_pd[column] = digitalize(target_meta_file_train_source_pd.loc[:, column])
    target_meta_file_train_pd = pd.DataFrame(target_meta_file_train_pd)
    return target_meta_file_train_pd


def adding_unique_counts(target_meta_file_train_pd):
    print("Adding unique counts...")
    stats = target_meta_file_train_pd.apply(lambda x: len(set(x)), axis=0)
    stats.name = '-1'
    target_meta_file_train_pd = target_meta_file_train_pd.append(stats, ignore_index=False)
    return target_meta_file_train_pd

def main():
    args = set_args()
    # load hierarchy file of wordnet
    hierarchy = load_hierarchy(args)
    parents = load_pc_pairs(args)
    # print(parents)
    double_parents = {i: parents[i] for i in parents if len(parents[i]) > 1}
    for i in double_parents:
        print(f"{i}:{double_parents[i]}")
    # print(f"dobule parent {len(double_parents)}")
    print(f"total nodes {len(parents)}")
    # find the max level
    max_level = find_max_level(hierarchy, max_level=0)
    print(f"Max Level {max_level}")

    # imagenet target labels
    imgnet_labels = load_imgnet_labels(args)

    # find class_stats
    class_stats = load_class_stats(args)

    # find usable classes 
    usable_classes, out_relationship_all_level = select_subset_candidate(hierarchy, imgnet_labels, args)
    usable_classes_pd = pd.DataFrame({'usable': usable_classes})

    for use_class in usable_classes:
        assert use_class in imgnet_labels
    
    
    # select 100 class for learning
    seed = 84
    np.random.seed(seed)
    target_classes = list(usable_classes_pd.iloc[np.random.permutation(np.arange(len(usable_classes)))[:100], 0])
    print("target_classes", target_classes)
    assert len(target_classes) == 100
    assert len(set(target_classes)) == 100
    print("unique target classes", len(set(target_classes)))

    # build metafile
    class_path_pd = load_image_path_class_pd(args)
    target_meta_file_train_source_pd = build_class_meta_data_source(class_path_pd, target_classes, out_relationship_all_level)
    meta_file_train_source_filename = os.path.join(args.root, args.meta_file_train_target100_source)
    target_meta_file_train_source_pd.to_csv(meta_file_train_source_filename)
    print(f"Saved target_meta_file_train_source_pd in {meta_file_train_source_filename}.")
    # digitalize
    target_meta_file_train_pd = build_digitalized_meta_data(target_meta_file_train_source_pd)
    # adding unique count
    target_meta_file_train_pd = adding_unique_counts(target_meta_file_train_pd)
    print(target_meta_file_train_pd)
    meta_file_train_filename = os.path.join(args.root, args.meta_file_train_target100)
    target_meta_file_train_pd.to_csv(meta_file_train_filename)
    print(f"Saved target_meta_file_train_pd in {meta_file_train_filename}.")
    


    

    


if __name__ == '__main__':
    main()
