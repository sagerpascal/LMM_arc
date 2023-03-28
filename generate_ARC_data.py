from arcdsl.main import get_data
import numpy as np
import json



def data_to_string(data, key):
    sample = {}
    sample['instruction'] = 'Please find the correct transformation of the test table based on the provided examples.'
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

    prepared = []
    train_keys = list(data['train'].keys())
    for key in train_keys:
        prepared.append(data_to_string(data, key))

    out_file = open('ARC_data.json', 'w')
    json.dump(prepared, out_file, indent=6)
    out_file.close()







if __name__ == '__main__':
    main()