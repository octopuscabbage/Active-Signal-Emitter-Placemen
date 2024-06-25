import matplotlib
matplotlib.use('Agg')

from smallab.utilities.experiment_loading.experiment_loader import experiment_iterator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from statsmaker import compute_statistics_nonparametric

def rename_trigger(trigger):
    if "wait" in trigger[0]:
        return f"wait {trigger[1]} {' '.join(map(str,trigger[2]))}"
    if isinstance(trigger,list):
        return " ".join(map(str,trigger))
    else:
        return str(trigger)

name = "test_minimizers"
save_load = False

hue_name = "Minimizer"

split_name = "Environment Seed"

boxplot = True
split_plot = False

rows = []
for experiment in experiment_iterator(name, use_tqdm=True):
    if experiment["result"] != []:
        if (boxplot and experiment["result"]["used_budget"]-1 != experiment["specification"]["planning_steps"]):
            continue
        #if experiment["specification"]["environment_seed"] in [4,5]:
        #    continue
        else:
            try:
                row = dict()
                row["Step"] = int(experiment["specification"]["budget"])
                row[hue_name] = str(experiment["specification"]["cem.optimizer"])
                row["Environment Seed"] = experiment["specification"]["environment_seed"]
                row["Objective Function"] = experiment["specification"]["objective_function"]
                row["Error"] = float(experiment["result"]["gt_lighting_rmse"])
                row["Seed"] = experiment["specification"]["seed"]
                row["Unit"] = str(row["Seed"]) + "_" + str(row["Environment Seed"])
                rows.append(row)
            except TypeError as e:
                print(e)
            except KeyError as e:
                print(e)

df = pd.DataFrame(rows)
print(df)
#print(df.dtypes)

#pairing_vars = ["Environment Seed", "Seed"]
#baselines = ["additive_factor_graph", "additive_gaussian_process","conditional_factor_graph"]
#for baseline in baselines:
#    for test in baselines:
#        if test != baseline:
#            stats = compute_statistics_nonparametric(df,hue_name,test,baseline,pairing_vars,"Error")
#            print(f"{test},  {baseline} :: {stats}")
if split_plot:
    fig, axes = plt.subplots(2,2)
    fig.suptitle("minimizer")
    iterator = zip(np.sort(df[split_name].unique()),axes.flatten())
else:
    plt.figure()
    plt.title("minimizer")
    iterator  = [(0,plt.gca())]
 
for split,ax in iterator:

#if True:
    #cur_df = df


    if split_plot:
        cur_df = df[(df[split_name] == split)].sort_values([hue_name])
        ax.set_title(f"{split_name}: {split}")
    else:
        cur_df = df

    print(cur_df)

    #plt.title(f"{split_name} {split}")
    #plt.title("Models")

    if boxplot:
        sns.boxplot(data=cur_df,y=hue_name,x="Error",ax=ax,showfliers=False)
    else:
        sns.lineplot(data=cur_df,x="Step",y="Error",  hue=hue_name,units="Unit",estimator=None)
plt.tight_layout()
    #plt.savefig(f"{name}_{split_name}_{split}.png")
plt.savefig("minimizer.png")


