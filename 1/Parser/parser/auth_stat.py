import csv
from os import listdir
from os.path import isfile, join

__author__ = 'Stranger'

delimiter = u'_soc_graph'
degree_pattern = u'_soc_graph_clean_raw.tsv'
degree_sent_pattern = u'_soc_graph_sentim_raw.tsv'
num_threads_pattern = u'_threads_count.tsv'
betweenness_pattern = u'_betweenness.tsv'
betweenness_sent_pattern = 'sentim'
prefix = 'auth_stat_using_'


def check_betweenness_file_exists(betweenness_files, f):
    part = f.partition(delimiter)
    name = part[0]

    for b in betweenness_files:
        bname = b.partition(delimiter)[0]
        if bname == name:
            return b

    raise ValueError(u'Matching betweenness file for ' + f + u' not found!')


def check_clean_graph_file_exists(clean_graph_files, f):
    part = f.partition(delimiter)
    name = part[0]

    for b in clean_graph_files:
        bname = b.partition(delimiter)[0]
        if bname == name:
            return b

    raise ValueError(u'Matching clean graph file for ' + f + u' not found!')


def check_sentim_graph_file_exists(sentim_graph_files, f):
    part = f.partition(delimiter)
    name = part[0]

    for b in sentim_graph_files:
        bname = b.partition(delimiter)[0]
        if bname == name:
            return b

    raise ValueError(u'Matching sentiment graph file for ' + f + u' not found!')


def check_threads_count_file_exists(sentim_graph_files, f):
    part = f.partition(delimiter)
    name = part[0]

    for b in sentim_graph_files:
        bname = b.partition(num_threads_pattern)[0]
        if name.startswith(bname):
            return b

    raise ValueError(u'Matching sentiment graph file for ' + f + u' not found!')


def merge_data(betweenness_sent_file, betweenness_file, clean_graph_file, sentim_graph_file, threads_count_file):
    reply_count_data = read_csv(clean_graph_file)
    reply_sentim_data = read_csv(sentim_graph_file)
    threads_count_data = read_csv(threads_count_file)[1:]
    betweenness_data = read_csv(betweenness_file)
    betweenness_sent_data = read_csv(betweenness_sent_file)

    authors = [x[0] for x in threads_count_data]
    betweenness = [x[1] for x in betweenness_data]
    in_degree = [sum([float(x) for x in row[1:]]) for row in list(zip(*reply_count_data))[1:]]
    out_degree = [sum([float(x) for x in row[1:]]) for row in reply_count_data[1:]]
    betweenness_sent = [x[1] for x in betweenness_sent_data]
    in_degree_sent = [sum([float(x) for x in row[1:]]) for row in list(zip(*reply_sentim_data))[1:]]
    out_degree_sent = [sum([float(x) for x in row[1:]]) for row in reply_sentim_data[1:]]
    threads_count = [x[1] for x in threads_count_data]

    result = list(zip(authors, betweenness, in_degree, out_degree, betweenness_sent,
                      in_degree_sent, out_degree_sent, threads_count))

    return result


def main():
    files = [f for f in listdir('.') if isfile(join('.', f))]
    betweenness_files = [f for f in files if f.endswith(betweenness_pattern) and betweenness_sent_pattern not in f]
    betweenness_sent_files = [f for f in files if f.endswith(betweenness_pattern) and betweenness_sent_pattern in f]
    clean_graph_files = [f for f in files if f.endswith(degree_pattern)]
    sentim_graph_files = [f for f in files if f.endswith(degree_sent_pattern)]
    threads_count_files = [f for f in files if f.endswith(num_threads_pattern)]

    for file in betweenness_sent_files:
        betweenness_file = check_betweenness_file_exists(betweenness_files, file)
        clean_graph_file = check_clean_graph_file_exists(clean_graph_files, file)
        sentim_graph_file = check_sentim_graph_file_exists(sentim_graph_files, file)
        threads_count_file = check_threads_count_file_exists(threads_count_files, file)

        result = merge_data(file, betweenness_file, clean_graph_file, sentim_graph_file, threads_count_file)
        result.insert(0, ['author', 'betweenness', 'in_degree', 'out_degree', 'betweenness_sent',
                          'in_degree_sent', 'out_degree_sent', 'num_of_threads'])

        part = file.partition(delimiter)
        filename = prefix + part[0] + part[2]

        print_to_csv(result, filename)


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


if __name__ == '__main__':
    main()
