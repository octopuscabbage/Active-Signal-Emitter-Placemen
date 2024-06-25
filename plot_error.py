from smallab.utilities.experiment_loading.experiment_loader import experiment_iterator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def rename_trigger(trigger):
    if "wait" in trigger[0]:
        return f"wait {trigger[1]} {' '.join(map(str,trigger[2]))}"
    if isinstance(trigger,list):
        return " ".join(map(str,trigger))
    else:
        return str(trigger)

name = "test_objective_pilot_isam"
save_load = False

style_name = "Objective"

hue_name = "Trigger"
#split_name = "Environment Seed"

boxplot = True

rows = []
for experiment in experiment_iterator(name, use_tqdm=True):
    if experiment["result"] != []:
        if (boxplot and experiment["result"]["used_budget"]-1 != experiment["specification"]["planning_steps"]):
            continue
        else:
            try:
                row = dict()
                row["Step"] = experiment["specification"]["budget"]
                row[hue_name] = rename_trigger(experiment["specification"]["lighting_trigger"])
                 
                row[style_name] = str(experiment["specification"]["objective_function"])

                row["Environment Seed"] = experiment["specification"]["environment_seed"]
                row["Objective Function"] = experiment["specification"]["objective_function"]
                row["Error"] = experiment["result"]["gt_lighting_rmse"]

                row["Seed"] = experiment["specification"]["seed"]
                rows.append(row)
            except TypeError as e:
                print(e)
            except KeyError as e:
                print(e)

df = pd.DataFrame(rows)
print(df)
    #cur_df = df[(df[split_name] == split) & (df["Objective Function"] == objective_function)]
#cur_df = cur_df.sort_values(by=[style_name,hue_name])
#if not cur_df.empty:
#for split in df[split_name].unique():
if True:
    #cur_df = df[(df[split_name] == split)].sort_values([hue_name,style_name])
    cur_df = df.sort_values([hue_name,style_name])
    #split_name = "All" 
    #split = ""
    plt.figure()

    #title = f"{split} - {objective_function}"
    #title = "Test"
    plt.title(f"{split_name} {split}")

    if boxplot:
        #sns.boxplot(data=cur_df,x=hue_name,y="Error",hue=style_name)

        sns.boxplot(data=cur_df,y=hue_name,x="Error",hue=style_name)
        # sns.stripplot(x=hue_name, y="Error", hue=style_name,
        #     data=cur_df, jitter=True)
    else:
        ax = plt.gca()
        sns.lineplot(data=cur_df, x="Step", y="Error",  hue=hue_name, ax=ax, style=style_name)
    plt.tight_layout()
    plt.savefig(f"{name}_{split_name}_{split}.png")
        #sns.lineplot(data=df, x="Step", y="Error", style="Seed", hue=hue_name,units="Seed",estimator=None,ax=ax)

#plt.savefig("figs/" + title.replace(".","").replace("/","") + ".png")
#plt.show()
