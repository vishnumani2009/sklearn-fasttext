# Set encoding to support Python 2
# -*- coding: utf-8 -*-

def read_file(filename):
    default_args = {}
    with open(filename, 'r') as f:
        for line in f:
            try:
                line = line.decode('utf-8')
            except:
                line = line

            raw = line.split(' ')
            key = raw[0]
            value = raw[1].strip()
            default_args[key] = value
    return default_args
