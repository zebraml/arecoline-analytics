import pandas as pd
import numpy as np

import argparse
import json

import matplotlib.pyplot as plt

def main(filepath, outpath, segment_size):
    divisor = (60 / segment_size)

    data = pd.read_csv(filepath, index_col=0)

    # detect minute from a segment #. Because we know segment size, we can compute number of segments per minute.
    data['Minute'] = (data['Segment_#'] / divisor).astype(int) + 1 

    n_classes = len(data['Class'].unique())
    axes_number = int(np.ceil(np.sqrt(n_classes)))
    
    fig, axes = plt.subplots(figsize=(4*axes_number, 3*axes_number), nrows=axes_number, ncols=axes_number)

    for idx, ((group_name, group), ax) in enumerate(zip(data.groupby('Class'), axes.flatten())):
        predictions = group.groupby(['Minute', 'Prediction']).size().unstack() 
        # divide by total amount of predictions to  convert count to proportion
        predictions = predictions / np.nansum(predictions, axis=1, keepdims=True)
        
        predictions.plot(kind='bar', stacked=True, ax=ax, legend=False)
        
        ax.set_title(group_name)
        ax.tick_params(rotation=0)

        if idx == 0:
            ax.legend(loc=(-0.5, 0.1))

    # disable excess axes
    for ax in axes.flatten()[idx + 1:]:
        ax.set_axis_off()
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect clusters in confusion matrix')

    parser.add_argument('--filepath', required=True, help='Path to a file with predictions')
    parser.add_argument('--outpath', required=True, help='Name of the out file with figure')
    parser.add_argument('--segment_size', type=float, default=30, help='Length of a segment in seconds (default=30)')

    args = parser.parse_args()

    main(args.filepath, args.outpath, args.segment_size)
