import sys

if sys.version_info < (3, 0):
    raise Exception('must use python 3.0 or greater')

import os
import csv
from os import listdir
from os.path import isfile, join

__author__ = 'Stranger'

auth_stat_pattern = 'auth_stat_'
stat_post_pattern = 'stat_'
prefix = 'total_'


def check_stat_file_exists(stat_files, f):
    for b in stat_files:
        bname, extension = os.path.splitext(b)
        bname = bname.partition('/')[2]
        if bname in f:
            return b

    raise ValueError(u'Matching sentiment graph file for ' + f + u' not found!')


def merge_data(auth_file, stat_file):
    auth_data = read_csv(auth_file)[1:]
    stat_data = read_csv(stat_file)[1:]

    auth_index = {x: i for i, x in enumerate([x[0] for x in auth_data])}
    merged = [list(x)[:-2] + auth_data[auth_index.get(x[-1])][1:] + list(list(x)[-2]) for x in stat_data]

    return merged


def read_csv(file):
    data = []
    with open(file, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        for row in csv_reader:
            data.append(row)

    return data


def print_to_csv(data, file):
    with open(file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        for row in data:
            csvwriter.writerow(list(row))


def main():
    directories = [f for f in listdir('.') if not isfile(join('.', f)) and f.startswith('keywords')]

    if not directories:
        raise ValueError('No one input directory found!')

    for directory in directories:
        print('Merge results for ' + directory)

        auth_files = [directory + '/' + f for f in listdir(directory) if
                      isfile(join(directory, f)) and f.startswith(auth_stat_pattern)]
        stat_files = [directory + '/' + f for f in listdir(directory) if
                      isfile(join(directory, f)) and f.startswith(stat_post_pattern)]

        if not auth_files:
            raise ValueError('No one input file found!')

        for auth_file in auth_files:
            print('    Merge results for ' + auth_file)

            stat_file = check_stat_file_exists(stat_files, auth_file)

            result = merge_data(auth_file, stat_file)
            result.insert(0, ['length', 'links', 'quoted', 'position',
                              'sentiment', 'keywords', 'betweenness',
                              'in_degree', 'out_degree', 'betweenness_sent', 'in_degree_sent',
                              'out_degree_sent', 'num_of_threads', 'utility'])

            name = stat_file.partition('/')[2].partition(stat_post_pattern)[2]
            filename = directory + '/' + prefix + name

            print_to_csv(result, filename)


if __name__ == '__main__':
    main()
