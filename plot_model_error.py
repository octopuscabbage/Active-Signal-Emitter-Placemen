import matplotlib
matplotlib.use('Agg')

from smallab.utilities.experiment_loading.experiment_loader import experiment_iterator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from statsmaker import compute_statistics_nonparametric
plt.rcParams["text.usetex"]


def rename_trigger(trigger):
    if "wait" in trigger[0]:
        return f"wait {trigger[1]} {' '.join(map(str,trigger[2]))}"
    if isinstance(trigger,list):
        return " ".join(map(str,trigger))
    else:
        return str(trigger)

def rename_model(model):
    if model == "conditional_factor_graph":
        return "Factor Graph"
    elif model == "additive_factor_graph":
        return "Residual Factor Graph"
    elif model == "additive_gaussian_process":
        return "Residual\nGaussian Process"
    else:
        raise Exception()

name = "test_model_new_intensity"
save_load = False

hue_name = "Model"

split_name = "Environment Seed"

boxplot = True
split_plot = False

rows = []
for experiment in experiment_iterator(name, use_tqdm=True):
    if experiment["result"] != []:
        if (boxplot and experiment["result"]["used_budget"]-1 != experiment["specification"]["planning_steps"]):
            continue
        if experiment["specification"]["environment_seed"] in [3]:
            continue
        else:
            try:
                row = dict()
                row["Step"] = int(experiment["specification"]["budget"])
                row[hue_name] = rename_model(experiment["specification"]["data_model"])
                if row[hue_name] == "Residual Factor Graph":
                    continue
                row["raw_name"] = str(experiment["specification"]["data_model"])
                row["Environment Seed"] = experiment["specification"]["environment_seed"]
                row["Objective Function"] = experiment["specification"]["objective_function"]
                row["Error"] = float(experiment["result"]["gt_lighting_rmse"])
                row["Seed"] = experiment["specification"]["seed"]
                rows.append(row)
            except TypeError as e:
                print(e)
            except KeyError as e:
                print(e)

df = pd.DataFrame(rows)
print(df)
#print(df.dtypes)

pairing_vars = ["Environment Seed", "Seed"]
baselines = ["additive_gaussian_process"]
test = "conditional_factor_graph"
for baseline in baselines:
        stats = compute_statistics_nonparametric(df,"raw_name",test,baseline,pairing_vars,"Error")
        print(f"{test},  {baseline} :: {stats}")
if split_plot:
    fig, axes = plt.subplots(2,2)
    fig.suptitle("model")
    iterator = zip(np.sort(df[split_name].unique()),axes.flatten())
else:
    plt.figure()
    #plt.title("Models")
    iterator  = [(0,plt.gca())]
 
for split,ax in iterator:

#if True:
    #cur_df = df


    if split_plot:
        cur_df = df[(df[split_name] == split)].sort_values([hue_name])
        ax.set_title(f"{split_name}: {split}")
    else:
        cur_df = df.sort_values([hue_name])

    print(cur_df)

    #plt.title(f"{split_name} {split}")
    #plt.title("Models")

    if boxplot:
        sns.boxplot(data=cur_df,y=hue_name,x="Error",ax=ax,showfliers=False)
    else:
        sns.scatterplot(data=cur_df,x="Step",y="Error",  hue=hue_name)
plt.tight_layout()
    #plt.savefig(f"{name}_{split_name}_{split}.png")
plt.savefig("models.png")
plt.savefig("models.pdf")


