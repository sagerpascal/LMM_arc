from arcdsl.main import get_data
import json
from pathlib import Path
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--add_descriptions', action='store_true', default=False, help='Add human descriptions to tasks')
args = parser.parse_args()

PATH_LARC_DATASET_BASE = Path("LARC/dataset")
PATH_LARC_SUMMARY = PATH_LARC_DATASET_BASE / "summary"


def get_larc_data():
    build_csv = pd.read_csv(str(PATH_LARC_SUMMARY / "build.csv"))
    description_csv = pd.read_csv(str(PATH_LARC_SUMMARY / "description.csv"))
    join_csv = pd.read_csv(str(PATH_LARC_SUMMARY / "join.csv"))
    task_csv = pd.read_csv(str(PATH_LARC_SUMMARY / "task.csv"))

    merged_csv = pd.concat([build_csv, description_csv, join_csv, task_csv], axis=1)
    merged_csv = merged_csv[merged_csv.is_success == True]
    result = merged_csv[["description_input", "description_output", "task_name"]]
    result["task_name"] = result["task_name"].str.replace(".json", "")
    return result


def get_task_description(task_descriptions, task_name):
    if task_name in task_descriptions.task_name.values:
        descr_in = task_descriptions[task_descriptions.task_name == task_name].description_input.values
        descr_out = task_descriptions[task_descriptions.task_name == task_name].description_output.values

        descr = ""
        for i, (di, do) in enumerate(zip(descr_in, descr_out)):
            if i > 0:
                descr += "\n or described in other words:\n"
            descr += di.replace("...", ":") + " ==> " + do.replace("...", ":")

        return descr
    else:
        return None


def data_to_string(data, key, descr=None):
    sample = {}
    sample['instruction'] = 'Please find the correct transformation of the test table based on the provided examples.'
    if descr is not None:
        sample['description'] = descr

    sample['input'] = ''

    for i, tr in enumerate(data['train'][key]):
        sample['input'] += ('Example ' + str(i) + ':' + '\n')
        sample['input'] += (str(tr['input']) + '\n')
        sample['input'] += ('==>' + '\n')
        sample['input'] += (str(tr['output']) + '\n')

    sample['input'] += ('Test:' + '\n')
    sample['input'] += (str(data['test'][key][0]['input']) + '\n')
    sample['input'] += ('==>' + '\n')
    sample['input'] += ('?' + '\n')

    sample['output'] = str(data['test'][key][0]['output'])

    return sample


def main():
    data = get_data(base_path='./data/')
    task_descriptions = get_larc_data()

    prepared = []
    train_keys = list(data['train'].keys())
    for key in train_keys:
        descr = get_task_description(task_descriptions, key) if args.add_descriptions else None
        prepared.append(data_to_string(data, key, descr=descr))

    out_file = open('ARC_data.json', 'w')
    json.dump(prepared, out_file, indent=6)
    out_file.close()


if __name__ == '__main__':
    main()
