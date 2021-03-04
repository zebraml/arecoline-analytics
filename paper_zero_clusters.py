
import pandas as pd
import networkx as nx 

import community as community_louvain

from collections import defaultdict, Counter

from sklearn.metrics import confusion_matrix

import argparse
import json

labels_order = ['AP', 'Arecoline', 'Caffeine', 'Cocaine', 'Ethanol',
       'Ibogaine', 'Ketamine', 'LSD', 'MDMA', 'Mescaline', 'Nicotine', 'PCP',
       'Psilocybin', 'THC', 'Control']

def get_the_most_frequent(values):
    class_to_counts = Counter(values)

    counts_to_class = defaultdict(list)

    for key, value in class_to_counts.items():
        counts_to_class[value].append(key)

    max_count = max(counts_to_class.keys())
        
    target = sorted(counts_to_class[max_count])[~0]
    
    return target


def transform_predictions(predictions):
    grouped = predictions.groupby('Fish_#')
    
    predictions_by_fish = {'Prediction': grouped['Prediction'].apply(get_the_most_frequent),
                           'Class': grouped['Class'].apply(get_the_most_frequent)}
    return pd.DataFrame(predictions_by_fish)


def main(filepath, outpath, resolution):
    predictions = pd.read_csv(filepath, index_col=0)
    predictions_fish = transform_predictions(predictions)

    cm = confusion_matrix(predictions_fish['Class'], predictions_fish['Prediction'], 
                      labels=labels_order, normalize='true')

    G = nx.Graph(cm.T)
    partition = community_louvain.best_partition(G, random_state=42, resolution=resolution)

    comm_to_idx = defaultdict(list)

    for idx, v in partition.items():
        comm_to_idx[v].append(labels_order[idx])
        
    json.dump(comm_to_idx, open(outpath, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect clusters in confusion matrix')

    parser.add_argument('--filepath', required=True, help='Path to a file with confusion matrix')
    parser.add_argument('--outpath', required=True, help='Name of the out file')
    parser.add_argument('--resolution', type=float, default=1.25, help='Resolution parameter for the Louvain method')

    args = parser.parse_args()

    main(args.filepath, args.outpath, args.resolution)
