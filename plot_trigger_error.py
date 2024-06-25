from smallab.utilities.experiment_loading.experiment_loader import experiment_iterator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from copy import deepcopy
from itertools import product

plt.rcParams["text.usetex"]

def rename_trigger(trigger):
    if "wait" in trigger[0]:
        return f"wait {trigger[1]} {' '.join(map(str,trigger[2]))}"
    if trigger[0] == "every":
        return "$n=1$"

    if trigger[0] == "last":
        return "Last Step"
    if trigger[0] == "every_n":
        if trigger[1] > 100:
            return "First Step"
        if trigger[1] == 10:
            return f"***$n={trigger[1]}$"

        else: 
            return f"$n={trigger[1]}$"
        #return f"Every $n$ steps, $n=$"
    if trigger[0] == "logprob_fraction":
        if trigger[1] == 1.1:
            return r"***$\alpha=" + str(trigger[1])+ "$"

        else:
            return r"$\alpha=" + str(trigger[1])+ "$"
    if isinstance(trigger,list):
        return " ".join(map(str,trigger))
    else:
        return str(trigger)

name = "test_triggers_new_intensity"
save_load = False

hue_name = "Trigger"

split_name = "Environment Seed"

boxplot = True
plot_cumulative = False
split_plot = False

cumulative_logprobs = [1.05,1.1,1.2]

rows = []
for experiment in experiment_iterator(name, use_tqdm=True):
    if experiment["result"] != []:
        trigger = experiment["specification"]["lighting_trigger"]
        try:
            row = dict()
            row["Step"] = int(experiment["specification"]["budget"])
            #row[hue_name] = rename_trigger(experiment["specification"]["lighting_trigger"])
            row["Max Steps"] = experiment["specification"]["planning_steps"]
             
            row[hue_name] = rename_trigger(experiment["specification"]["lighting_trigger"])
            row["Param"] = experiment["specification"]["lighting_trigger"][1] if len(experiment["specification"]["lighting_trigger"]) > 1 else ""
            row["Method"] = "Logprob Fraction" if "logprob_fraction" == experiment["specification"]["lighting_trigger"][0] else "Every $n$"
            if trigger[0] not in ["logprob_fraction","every_n","every"]:
                row["Type"] = "Other"
            elif trigger[0] == "logprob_fraction":
                row["Type"] = "Logprob"
            elif trigger[0] in ["every_n","every"]:
                if trigger[0] == "every_n" and trigger[1] > 100:
                    row["Type"] = "Other"
                else:
                    row["Type"] = "Every $n$"

            if plot_cumulative:
                if row["Type"] == "Other" or trigger[0] == "every":
                    continue
                if trigger[0] == "logprob_fraction":
                    if trigger[1] not in cumulative_logprobs:
                        continue
                if trigger[0] == "every_n" and trigger[1] < 5:
                    continue

            row["sort_key"] = experiment["specification"]["lighting_trigger"]

            row["Environment Seed"] = experiment["specification"]["environment_seed"]
            row["Objective Function"] = experiment["specification"]["objective_function"]
            row["Error"] = float(experiment["result"]["gt_lighting_rmse"])
            row["Replaced Lights"] = 1 if experiment["result"]["replaced_lights"] else 0

            row["Seed"] = experiment["specification"]["seed"]
            rows.append(row)
        except TypeError as e:
            print(e)
        except KeyError as e:
            print(e)

df = pd.DataFrame(rows)


#for row in df:
#    cumulative[("Environment Seed", "Seed", hue_name)] += 1

if plot_cumulative:
    df = df.sort_values(["Step"])
    df["Cumulative Triggers"] = df.groupby(["Environment Seed","Seed",hue_name])["Replaced Lights"].cumsum()
    print(df)
    added_rows = []
    #we only log on replanning so we need to interpolate the values
    for environment_seed, seed, trigger in tqdm(list(product(list(df["Environment Seed"].unique()), list(df["Seed"].unique()), list(df[hue_name].unique())))):
        cur_df = df.loc[(df["Environment Seed"] == environment_seed) & (df["Seed"] == seed) & (df[hue_name] == trigger)]
        last_cumulative_triggers = None
        last_step = None
        for _,row in cur_df.sort_values("Step").iterrows():
            if last_cumulative_triggers is None:
                last_cumulative_triggers = row["Cumulative Triggers"]
                last_step = 0
            else:
                for i in range(last_step+1, row["Step"]):
                    new_row = deepcopy(row)
                    new_row["Step"] = i
                    new_row["Cumulative Triggers"] = last_cumulative_triggers
                    added_rows.append(new_row)
                last_step = row["Step"]
                last_cumulative_triggers = row["Cumulative Triggers"]
    added = pd.DataFrame(added_rows)
    df = pd.concat([df,added],ignore_index=True)
    print(df)


#f["Cumulative Triggers"] = list(cumulative.values())
#for split in df[split_name].unique():
#    cur_df = df[(df[split_name] == split)].sort_values([hue_name])

if split_plot:
    fig, axes = plt.subplots(3,2)
    #fig.suptitle("model")
    iterator = zip(np.sort(df[split_name].unique()),axes.flatten())
else:
    plt.figure()
    #plt.title("Models")
    iterator  = [(0,plt.gca())]

for split,ax in iterator:

#if True:
    #cur_df = df


    if split_plot:
        cur_df = df[(df[split_name] == split)].sort_values(["sort_key"])
        ax.set_title(f"{split_name}: {split}")
    else:
        cur_df = df.sort_values(["sort_key"])
    #cur_df = df.sort_values([hue_name])
    #print(cur_df)
    #plt.title(f"{split_name} {split}")

    if boxplot:
        bp_df = cur_df.loc[df["Step"]-1 == df["Max Steps"]]

        if plot_cumulative:
            sns.boxplot(data=bp_df,y=hue_name,hue="Type",x="Cumulative Triggers",ax=ax)
        else:
            ax = sns.boxplot(data=bp_df,y=hue_name,hue="Type",x="Error",ax=ax,showfliers=False,dodge=False,whis=1.2)
            sns.move_legend(ax,"lower right")
    else:
        if plot_cumulative:
            sns.lineplot(data=cur_df,x="Step",y="Cumulative Triggers", hue=hue_name,ax=ax,style="Method")
            #sns.lmplot(data=cur_df,x="Step",y="Cumulative Triggers", hue=hue_name,x_bins=bins,x_ci="ci",fit_reg=False)
        else:
            sns.scatterplot(data=cur_df,x="Step",y="Error",  hue=hue_name,ax=ax)
plt.tight_layout()
#plt.savefig(f"{name}_{split_name}_{split}.png")
if plot_cumulative:
    plt.savefig("triggers_cumulative.png")
    plt.savefig("triggers_cumulative.pdf")
else:
    plt.savefig("triggers.png")
    plt.savefig("triggers.pdf")


