import sys

if sys.version_info < (3, 0):
    raise Exception('must use python 3.0 or greater')

import os
import csv
from os import listdir
from os.path import isfile, join

__author__ = u'Stranger'

file_pattern = u'total_'
output_extension = u'arff'
determined_attributes = {
    # u'sentiment': [-2, -1, 0, 1, 2],
    # u'utility': [0, 1, 2, 3, 4, 5]
}

DATA = u'@DATA'
RELATION = u'@RELATION'
ATTRIBUTE = u'@ATTRIBUTE'
NUMERIC = u'NUMERIC'


def read_csv(input_file):
    input_data = []

    with open(input_file, u'r', newline=u'') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=u'\t')
        for row in csv_reader:
            input_data.append(row)

    return input_data


def convert_file(file_name):
    input_data = read_csv(file_name)

    if len(input_data) < 2:
        raise ValueError(u'Input data must contains at least one row!')

    file_name, file_extension = os.path.splitext(file_name)
    header = input_data[0]
    input_data = input_data[1:]

    if len(header) < 2:
        raise ValueError(u'Data set must contains at least two attributes!')

    with open(file_name + u'.' + output_extension, u'w', newline=u'') as arff_file:
        arff_file.write(u'\n' + RELATION + u' ' + file_name + u'\n\n')

        for attribute in header:
            if attribute in determined_attributes:
                attribute_values = [str(i) for i in determined_attributes[attribute]]
                arff_file.write(ATTRIBUTE + u' ' + attribute + u' {' + u', '.join(attribute_values) + u'}\n')
            else:
                arff_file.write(ATTRIBUTE + u' ' + attribute + u' ' + NUMERIC + u'\n')

        arff_file.write(u'\n' + DATA + u'\n')

        for data_row in input_data:
            if not len(data_row) == len(header):
                raise ValueError(u'Each row must have the same count of records as presented ih header!')

            data = [str(i) for i in data_row]
            arff_file.write(u', '.join(data) + u'\n')


def main():
    directories = [f for f in listdir('.') if
                   not isfile(join('.', f)) and f.startswith('keywords')]

    if not directories:
        raise ValueError('No one input directory found!')

    for directory in directories:
        print('Convert files for ' + directory)

        total_files = [directory + '/' + f for f in listdir(directory) if
                       isfile(join(directory, f)) and f.startswith(file_pattern) and not f.endswith('.arff')]

        if not total_files:
            raise ValueError('No one input file found!')

        for file in total_files:
            print('    Convert file ' + file)
            convert_file(file)


if __name__ == u'__main__':
    main()
