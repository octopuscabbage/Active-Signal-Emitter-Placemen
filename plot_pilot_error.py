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

def rename_objective_function(obj):
    if obj[0] == "ucb_ambient":
        return r"$UCB_{r}, \beta=" + "{:0.3}".format(float(obj[1])) + "$"
    elif obj[0] == "ucb":
        return r"$UCB, \beta=" + "{:0.3}".format(float(obj[1])) + "$"
    elif obj[0] == "ei_ambient":
        return r"$EI_{r} , \xi=" + str(obj[1]) + "$"
    elif obj == "global_mi":
        return "Variance Reduction"
    elif obj == "entropy":
        return "Entropy"
    else:
        raise Exception(str(obj))
name = "test_objective_pilot"
save_load = False

hue_name = "Objective"

split_name = "Environment Seed"

boxplot = True
split_plots = False

rows = []
for experiment in experiment_iterator(name, use_tqdm=True):
    if experiment["result"] != []:
        if (boxplot and experiment["result"]["used_budget"]-1 != experiment["specification"]["planning_steps"]):
            continue
        #if experiment["specification"]["planning_steps"] != 75:
        #    continue
        obj_fn = experiment["specification"]["objective_function"]
        if False:#(isinstance(obj_fn,list) and obj_fn[1] in [0.01,10]):
            continue
        #if experiment["specification"]["environment_seed"] in [5]:
        #    continue
        #if isinstance(obj_fn,list) and ((obj_fn[0] == "ucb" and obj_fn[1] != 1.0) or (obj_fn[0] == "ucb_ambient" and obj_fn[1] != 0.01)):
                #continue
        else:
            try:
                row = dict()
                row["Step"] = float(experiment["specification"]["budget"])
                row[hue_name] = rename_objective_function(experiment["specification"]["objective_function"]) + " " + str(experiment["specification"]["do_pilot_survey"])
                
                row["is_best"] = (isinstance(obj_fn,list) and obj_fn[0] == "ucb_ambient" and obj_fn[1] == 0.05)

                row["Environment Seed"] = experiment["specification"]["environment_seed"]
                #row["Objective Function"] = str(experiment["specification"]["objective_function"]) + str(experiment["specification"]["rollout_strategy"])
                row["Error"] = float(experiment["result"]["gt_lighting_rmse"])

                of = experiment["specification"]["objective_function"]
                row["of_sort"] = str(list(of)) if isinstance(of,str) else str(of)

                row["Seed"] = float(experiment["specification"]["seed"])
                rows.append(row)
            except TypeError as e:
                print(e)
            except KeyError as e:
                print(e)

df = pd.DataFrame(rows)
#print(df)
print(df.dtypes)

#best_df = df.loc[df["is_best"] == True]
#print(best_df)
#best_median = best_df["Error"].median()

#for split in df[split_name].unique():
#    cur_df = df[(df[split_name] == split)].sort_values([hue_name])

pairing_vars = ["Environment Seed", "Seed"]
print(df.sort_values(pairing_vars))
#baselines = ["entropy", "global_mi", str(["ucb",1.0])]
#for baseline in baselines:
#    stats = compute_statistics_nonparametric(df,"Objective Function",str(["ucb_ambient",0.01]),baseline,pairing_vars,"Error")
#    print(f"{baseline} :: {stats}")
if split_plots:
    fig, axes = plt.subplots(3,1)
    fig.suptitle("Objective")

    iterable = zip(np.sort(df[split_name].unique()),axes.flatten())
else:
    iterable = [(0,0)]

for split,ax in iterable:
    
    if split_plots:
        cur_df = df[(df[split_name] == split)].sort_values(["of_sort"])
        ax.set_title(f"{split_name}: {split}")

        plt.title(f"{split_name} {split}")
    else:

        cur_df = df.sort_values(["of_sort"])
        plt.figure()

        plt.title("Objective")
        ax  = plt.gca()

    if boxplot:
        sns.boxplot(data=cur_df,y=hue_name,x="Error",ax=ax)
        sns.stripplot(data=cur_df,x="Error",y=hue_name,color="black",alpha=0.3,ax=ax)
        #handles, labels = ax.get_legend_handles_labels()
        #print(len(handles))

        #ax.legend(handles=[(handles[0], handles[2]), (handles[1], handles[3])],
        #                    loc='upper left', handlelength=4,
        #                    handler_map={tuple: HandlerTuple(ndivide=None)})
        #plt.axvline(x=best_median,linestyle="--",alpha=0.9,color="gray",ax=ax)
    else:
        #sns.scatterplot(data=cur_df,x="Step",y="Error",  hue=hue_name,ax=ax)
        sns.lineplot(data=cur_df,x="Step",y="Error", hue=hue_name, ax=ax)

plt.tight_layout()
    #plt.savefig(f"objective_{name}_{split_name}_{split}.png")
plt.savefig("objective_pilot.png")

