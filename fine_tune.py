
def data_to_string(data, key):
    data_string = ''
    for i, tr in enumerate(data['train'][key]):
        data_string += ('Example ' + str(i) + ':' + '\n')
        data_string += (tr['input'] + '\n')
        data_string += ('==>')
        data_string += (tr['output'])

    data_string += ('Test:')
    data_string += (data['test'][key][0]['input'])
    data_string += ('==>')
    data_string += ('?')

