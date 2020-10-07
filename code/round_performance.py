import experiment_manager as xpm
from pathlib import Path

import os
from os.path import join
import argparse

import pandas as pd

import altair as alt
from altair_saver import save

def generate_plot(data, name):
    data['server_val_accuracy'] = data['server_val_accuracy'][:-1]
    count = data['distill_acc'].shape[1] if data['distill_acc'].ndim == 2 else 1
    distill_acc = pd.DataFrame(data['distill_acc'], columns=[f'client{i}'for i in range(count)])
    distill_acc.index.name = "round"
    distill_acc = distill_acc.reset_index().melt('round')


    base = alt.Chart(distill_acc).transform_calculate(
        line="'Average'",
        shade="'Sigma1 confidence interval'",
    ).properties(
        title={
            'text': 'Rounds vs. Performance for One Step Distillation',
            'subtitle': name
        }
    )

    mean = base.mark_line().encode(
        x='round',
        y='mean(value)',
        color=alt.Color('line:N', scale=alt.Scale(range=['black']), legend=alt.Legend(symbolType="stroke"), title=""),
    )

    band = base.mark_errorband(extent='ci').encode(
        x='round',
        y=alt.Y('value', title='Accuracy on the test set'),
        color=alt.Color('shade:N', scale=alt.Scale(range=['steelblue']), legend=alt.Legend(symbolType="circle"), title=""),
    )



    plot = (mean + band).resolve_scale(
        color='independent'
    )
    return plot
    
parser = argparse.ArgumentParser(description='Generate plots from experiments')
parser.add_argument('sourcedir', default='./',
                   help='Path to where the experiments are stored')
parser.add_argument('targetdir', default='./',
                   help='Path to where the plots should be stored')

args = parser.parse_args()

currpath=os.getcwd()

#construct absolute path to result directory

source_path = join(currpath, args.sourcedir)
result_path = join(currpath, args.targetdir)

list_of_experiments = xpm.get_list_of_experiments(source_path)


for i in range(len(list_of_experiments)):
    data = list_of_experiments[i].results
    hyp = data['hyperparameters']
    del data['hyperparameters']
    hyp = list_of_experiments[i].hyperparameters
    name = f"{hyp['net']} - {hyp['dataset']} with {hyp['n_distill']} datapoints - {hyp['classes_per_client']}  - Phase: {hyp['distill_phase']} - Warmup: {hyp['warmup_type']} - % local data: {hyp['local_data_percentage']} - Distill weight: {hyp['distill_weight']}"
    save(generate_plot(data, name), join(result_path, name + '.svg'))
