import json
import os
import pprint

path = "/users/anup/tofino/bf-sde-8.9.2/build/p4-build/bench"

all_exps = {}
j = 0

def get_last_used_stage(stage_list):
    max_num = -1
    for stage in stage_list:
        if len(stage["xbar_bytes"]["bytes"]) > 0:
            max_num = stage["stage_number"]
    return max_num

def is_successful_map(subdir):
    #need = len(resources["resources"]["pipes"][0]["parser"]) > 0
    gen_cpp_dir = os.path.join(subdir, "../gen-cpp")
    return len(os.listdir(gen_cpp_dir)) > 2

summary_functions = {
    "stages": lambda x: len(x["xbar_bytes"]["bytes"]),
    "xbar_bytes": lambda x: len(x["xbar_bytes"]["bytes"]),
    "hash_distribution_units": lambda x: len(x["hash_distribution_units"]["units"]),
    "srams": lambda x: len(x["rams"]["srams"]),
    "map_rams": lambda x: len(x["map_rams"]["maprams"]),
    "meter_alus": lambda x: len(x["meter_alus"]["meters"]),
    "tcams": lambda x: len(x["tcams"]["tcams"])
}

for subdir, dirs, files in os.walk(path):
    if 'resources.json' in files:
        f = open(os.path.join(subdir, 'resources.json'))
        resources = json.load(f)
        f.close()

        exp_name = resources["program_name"]
        if(not is_successful_map(subdir)):
            #print(exp_name)
            all_exps[exp_name] = False
            continue

        stages = resources["resources"]["pipes"][0]["mau"]["mau_stages"]
        num_stages_used = get_last_used_stage(stages) + 1
        summary = {}
        overall_summary = {}
        for i in range(num_stages_used):
            stage = stages[i]
            stage_summary = {}
            for attribute, summary_function in summary_functions.items():
                stage_summary[attribute] = summary_function(stage)
                overall_summary[attribute] = overall_summary.get(attribute, 0) \
                                             + stage_summary[attribute]
            overall_summary["stages"] = num_stages_used
            summary[stage["stage_number"]] = stage_summary

        # pprint.pprint((summary, overall_summary))
        all_exps[exp_name] = (summary, overall_summary)

print("name, {}".format(", ".join(summary_functions.keys())))
for exp, data in all_exps.items():
    if data:
        print("{}, {}".format(exp, ", ".join(map(str, data[1].values()))))
    else:
        print("{}, ".format(exp))
