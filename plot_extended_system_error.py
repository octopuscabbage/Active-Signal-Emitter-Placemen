from smallab.utilities.experiment_loading.experiment_loader import experiment_iterator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from copy import deepcopy
from itertools import product
import matplotlib.ticker as mtick

from statsmaker import compute_statistics_nonparametric
from collections import defaultdict


plt.rcParams["text.usetex"]

def rename_trigger(trigger):
    if "wait" in trigger[0]:
        return f"wait {trigger[1]} {' '.join(map(str,trigger[2]))}"
    if trigger[0] == "every":
        return "Every Step"
    if trigger[0] == "last":
        return "Last Step"
    if trigger[0] == "every_n":
        if trigger[1] > 100:
            return "First Step"
        return f"Every {trigger[1]} steps"
    if trigger[0] == "logprob_fraction":
        return r"Logprob $\alpha=" + str(trigger[1]) + "$"
    if isinstance(trigger,list):
        return " ".join(map(str,trigger))
    else:
        return str(trigger)

name = "test_system_extended_new_intensity"
save_load = False

hue_name = "Method"

split_name = "Difficulty"

boxplot = True
plot_cumulative = False
split_plot = True

rows = []
for experiment in experiment_iterator(name, use_tqdm=True):
    if experiment["result"] != []:
        trigger = experiment["specification"]["lighting_trigger"]
        if experiment["specification"]["environment_seed"] in [3]:
            continue
        #if experiment["specification"]["lighting_trigger"] == ["every_n", 25]:
        #    continue
        #if experiment["specification"]["lighting_trigger"] == ["every"] and plot_cumulative:
        #    continue
        else:
            try:
                row = dict()
                row["Step"] = int(experiment["specification"]["budget"])
                #row[hue_name] = rename_trigger(experiment["specification"]["lighting_trigger"])
                row["Max Steps"] = experiment["specification"]["planning_steps"]
                 
                row["Method"] = "Proposed" if "logprob_fraction" == experiment["specification"]["lighting_trigger"][0] else "Baseline"
                ambient_lights = experiment["specification"]["ambient_lights"]
                if ambient_lights == 5:
                    row["Difficulty"] = "Easy"
                elif ambient_lights == 15:
                    row["Difficulty"] = "Medium"
                elif ambient_lights == 30:
                    row["Difficulty"] = "Hard"
                else:
                    raise Exception()

                row["sort_key"] = experiment["specification"]["lighting_trigger"]

                row["Environment Seed"] = experiment["specification"]["environment_seed"]
                row["Objective Function"] = experiment["specification"]["objective_function"]
                row["Error"] = float(experiment["result"]["gt_lighting_rmse"])
                row["Replaced Lights"] = 1 if experiment["result"]["replaced_lights"] else 0

                row["Seed"] = experiment["specification"]["seed"]
                row["match"] = str((experiment["specification"]["environment_seed"],row["Seed"],row["Difficulty"]))
                row["Environment"] = experiment["specification"]["environment_seed"]
                rows.append(row)
                
            except TypeError as e:
                print(e)
            except KeyError as e:
                print(e)

improvements = defaultdict(lambda: defaultdict(list))
improvement_rows = []
#for row in rows:
#    if row["Method"] == "Proposed" and row["Step"]-1 == row["Max Steps"]:
#        for row2 in rows:
#            if row2["Environment"] == row["Environment"] and row2["Difficulty" == row["Difficulty"] and row2["Method"] == "Baseline" and row2["Step"]-1 == row2["Max Steps"]:
#                match_count[row["match"]] += 1
#                starting_error = row2["Error"]
#                ending_error = row["Error"]
#                #percent_improvement = (starting_error - ending_error) / np.abs(starting_error) * 100
#                improvement = starting_error - ending_error
                #if improvement > 0:
                #improvement_rows.append({"Difficulty": row["Difficulty"],"Error Reduction": improvement})
for row in rows:
    if row["Step"]-1 == row["Max Steps"]:
        improvements[(row["Difficulty"], row["Environment"])][row["Method"]].append(row["Error"])

for key,errors in improvements.items():
    #print(errors["Baseline"])
    #print(errors['Proposed'])
    #print(np.median(errors["Baseline"]) - np.median(errors["Proposed"]))
    starting_error =np.median(errors["Baseline"])
    ending_error = np.median(errors["Proposed"])
    percent_improvement = (starting_error - ending_error) / np.abs(starting_error) * 100

    improvement_rows.append({"Difficulty": key[0], "Percent Error Reduction": percent_improvement})
improvement = pd.DataFrame(improvement_rows)



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

pairing_vars = ["Environment Seed", "Seed"]
baselines = ["Baseline"]
test = "Proposed"

#bp_df = df.loc[df["Step"]-1 == df["Max Steps"]]
#for baseline in baselines:
#        stats = compute_statistics_nonparametric(bp_df,"Method",test,baseline,pairing_vars,"Error")
#        print(f"{test},  {baseline} :: {stats}")


if split_plot:
    fig, axes = plt.subplots(3,1)
    #fig.suptitle("model")
    iterator = zip(np.sort(df[split_name].unique()),axes.flatten())
else:
    plt.figure()
    plt.title("Models")
    iterator  = [(0,plt.gca())]

order = ["Proposed","Baseline"]

for split,ax in iterator:

#if True:
    #cur_df = df
    

    if split_plot:
        #plt.figure()
        cur_df = df.loc[(df[split_name] == split)].sort_values(["sort_key"])
        #plt.title(f"{split_name}: {split}")
    else:
        cur_df = df.sort_values(["sort_key"])
    #cur_df = df.sort_values([hue_name])
    #print(cur_df)
    #plt.title(f"{split_name} {split}")

    if boxplot:
        bp_df = cur_df.loc[df["Step"]-1 == df["Max Steps"]]
        if plot_cumulative:
            sns.boxplot(data=cur_df,y=hue_name,x="Cumulative Triggers",ax=ax,order=order)
        else:
            if split_plot:
                print(bp_df)
                plt.figure()
                ax = plt.gca()
                sns.boxplot(data=bp_df,y="Method",x="Error",order=order,showfliers=False)
                plt.savefig(f"system_extended_{split}.png")
                plt.savefig(f"system_extended_{split}.pdf")

                plt.tight_layout()

            else:
                sns.boxplot(data=cur_df,y=hue_name,x="Error",ax=ax,order=order,showfliers=True)
    else:
        if plot_cumulative:
            sns.lineplot(data=cur_df,x="Step",y="Cumulative Triggers", hue=hue_name,ax=ax,style="Difficulty")
            #sns.lmplot(data=cur_df,x="Step",y="Cumulative Triggers", hue=hue_name,x_bins=bins,x_ci="ci",fit_reg=False)
        else:
            sns.scatterplot(data=cur_df,x="Step",y="Error",  hue=hue_name,ax=ax,order=order)
            
plt.tight_layout()
#plt.savefig(f"{name}_{split_name}_{split}.png")
if plot_cumulative:
    plt.savefig("system_extended_cumulative.png")
else:
    plt.figure()
    print(improvement)
    print(len(improvement_rows))
    order=["Easy","Medium","Hard"]
    sns.boxplot(data=improvement,y="Difficulty",x="Percent Error Reduction",showfliers=False,order=order)
    ax = plt.gca()
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    plt.savefig("system_improvement.png")
    plt.savefig("system_improvement.pdf")



