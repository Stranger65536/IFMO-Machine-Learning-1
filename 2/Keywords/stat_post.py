import sys

if sys.version_info < (3, 0):
    raise Exception('must use python 3.0 or greater')

import re
import os
import csv
from collections import Counter
from os import listdir
from os.path import isfile, join

__author__ = 'Stranger'

image_label = '**IMAGE**'
video_label = '**VIDEO**'
input_file = 'thread1.tsv'
output_name = 'post_1'
output_extension = '.tsv'
output_file = output_name + output_extension


def get_words_count(data):
    result = []

    for row in data:
        text = row[2]
        words = text.split()
        result.append(sum(Counter(words).values()))

    return result


def get_links_count(data):
    result = []

    for row in data:
        text = row[2]
        count = len(re.findall(image_label.replace('*', '\*') +
                               '|' + video_label.replace('*', '\*') +
                               '|' + 'http://', text))
        result.append(count)

    return result


def get_positions(data):
    return list(range(1, len(data) + 1))


def get_sentiment(data):
    result = []

    for row in data:
        result.append(row[5])

    return result


def get_utility(data):
    result = []

    for row in data:
        result.append(row[6])

    return result


def get_keywords_count(data, keywords):
    result = []
    pattern = '|'.join(keywords)

    for row in data:
        text = row[2]
        count = len(re.findall(pattern, text))
        result.append(count)

    return result


def find_all_replies(data, words):
    result = []

    for i in range(0, len(data)):
        text = data[i][2].split()
        joined_text = ''.join(text)
        joined_words = ''.join(words[0:3])

        if joined_words in joined_text:
            result.append(i)

    return result


def get_replied_posts(row, replied_authors, data):
    authors_left = replied_authors.copy()
    # noinspection PyUnusedLocal
    result = [-1 for x in replied_authors]

    for row in reversed(range(row)):
        author = data[row][1]

        if author in authors_left:
            authors_left.remove(author)
            result[replied_authors.index(author)] = row

    return result


def get_naive_reply_count(data, authors,
                          social_graph, social_graph_normalized,
                          social_graph_sentim, social_graph_sentim_normalized, authors_index):
    result = {}
    row_count = 1

    for row in range(0, len(data)):
        print(str(row_count) + ' of ' + str(len(data)) + ' rows processed', end="\r")
        row_count += 1
        text = data[row][2]

        replied_authors = list(set([x for x in authors if x in text]))
        replied_posts = get_replied_posts(row, replied_authors, data)

        for i in range(len(replied_authors)):
            if replied_posts[i] is not -1:
                replier = data[row][1]
                replier_id = authors_index[replier]
                replied_author = data[i][1]
                replied_author_id = authors_index[replied_author]

                reply_sentiment = int(data[row][5]) + 3

                social_graph[replier_id][replied_author_id] += 1
                social_graph_sentim[replier_id][replied_author_id] += reply_sentiment
                social_graph_sentim_normalized[replier_id][replied_author_id] += 1 / reply_sentiment

                count = result.get(replied_posts[i], 0)
                result[replied_posts[i]] = count + 1

    temp = []

    for row in range(0, len(social_graph)):
        for column in range(0, len(social_graph[row])):
            if int(social_graph[row][column]) is 0:
                social_graph_normalized[row][column] = sys.maxsize
            else:
                social_graph_normalized[row][column] = 1.0 / int(social_graph[row][column])

            if social_graph_sentim_normalized[row][column] is 0:
                social_graph_sentim_normalized[row][column] = sys.maxsize

    for i in range(0, len(data)):
        temp.append(result.get(i, 0))

    return temp


def get_reply_count(data, authors_pattern,
                    social_graph, social_graph_normalized,
                    social_graph_sentim, social_graph_sentim_normalized, authors_index):
    result = {}
    row_count = 1

    for row in data:
        print(str(row_count) + u' of ' + str(len(data)) + u' rows processed', end="\r")
        row_count += 1
        text = row[2]

        if 'Сообщение от' in text:
            reply_count = text.count('Сообщение от')

            if reply_count > 1:
                pass

            for i in range(0, reply_count):
                raw_reply = text.partition('Сообщение от')[2]
                reply = raw_reply.partition('Сообщение от')[0].strip()
                matches = re.match('(' + authors_pattern + ').*', reply)

                if matches:
                    author = matches.group(1)
                    message = reply.partition(author)[2]
                    words = message.split()
                    if len(words) < 3:
                        continue
                    else:
                        replies = find_all_replies(data, words)

                        if replies:
                            replier = row[1]
                            replier_id = authors_index[replier]
                            replied_post = data[replies[0]]
                            replied_author = replied_post[1]
                            replied_author_id = authors_index[replied_author]

                            reply_sentiment = int(row[5]) + 3

                            social_graph[replier_id][replied_author_id] += 1
                            social_graph_sentim[replier_id][replied_author_id] += reply_sentiment
                            social_graph_sentim_normalized[replier_id][replied_author_id] += 1 / reply_sentiment

                            count = result.get(replies[0], 0)
                            result[replies[0]] = count + 1

    temp = []

    for row in range(0, len(social_graph)):
        for column in range(0, len(social_graph[row])):
            if int(social_graph[row][column]) is 0:
                social_graph_normalized[row][column] = sys.maxsize
            else:
                social_graph_normalized[row][column] = 1.0 / int(social_graph[row][column])

            if social_graph_sentim_normalized[row][column] is 0:
                social_graph_sentim_normalized[row][column] = sys.maxsize

    for i in range(0, len(data)):
        temp.append(result.get(i, 0))

    return temp


def make_matrix_graph(social_graph, authors):
    # noinspection PyUnusedLocal
    extended = [[0 for x in range(len(social_graph) + 1)] for x in range(len(social_graph) + 1)]
    extended[0][0] = ''
    extended[0][1:] = authors[:]
    extended[1:][0] = authors[:]

    for i in range(0, len(authors)):
        extended[i + 1][1:] = social_graph[i][:]
        extended[i + 1][0] = authors[i]

    return extended


def get_threads_count(data, authors, authors_index):
    # noinspection PyUnusedLocal
    result = [0 for x in range(len(authors))]

    for row in data:
        author = row[1]
        index = authors_index[author]
        result[index] += 1

    temp = list(zip(authors, result))
    temp.insert(0, ('author', 'num_of_threads'))

    return temp


def get_post_stat(keywords_count, links_count, positions, reply_count, sentiment, utility, words_count, authors):
    post_statistics = list(zip(words_count, links_count,
                               reply_count, positions,
                               sentiment, keywords_count, utility, authors))
    post_statistics.insert(0, ('length', 'links',
                               'quoted', 'position',
                               'sentiment', 'keywords',
                               'utility', 'author'))
    return post_statistics


def naive_quoting(authors, authors_index, data, filename):
    # noinspection PyUnusedLocal
    naive_quoted_social_graph_clean_raw = [[0 for x in range(len(authors))] for x in range(len(authors))]
    # noinspection PyUnusedLocal
    naive_quoted_social_graph_clean_normalized = [[0 for x in range(len(authors))] for x in range(len(authors))]
    # noinspection PyUnusedLocal
    naive_quoted_social_graph_sentim_raw = [[0 for x in range(len(authors))] for x in range(len(authors))]
    # noinspection PyUnusedLocal
    naive_quoted_social_graph_sentim_normalized = [[0 for x in range(len(authors))] for x in range(len(authors))]

    print('    Calculate naive quoting')

    reply_count_naive_quoted = get_naive_reply_count(data, authors,
                                                     naive_quoted_social_graph_clean_raw,
                                                     naive_quoted_social_graph_clean_normalized,
                                                     naive_quoted_social_graph_sentim_raw,
                                                     naive_quoted_social_graph_sentim_normalized,
                                                     authors_index)

    print_to_csv(make_matrix_graph(naive_quoted_social_graph_clean_raw, authors),
                 filename + '/' + output_name + '_naive_quoted' + '_soc_graph_clean_raw' + output_extension)
    print_to_csv(make_matrix_graph(naive_quoted_social_graph_clean_normalized, authors),
                 filename + '/' + output_name + '_naive_quoted' + '_soc_graph_clean_normalized' + output_extension)
    print_to_csv(make_matrix_graph(naive_quoted_social_graph_sentim_raw, authors),
                 filename + '/' + output_name + '_naive_quoted' + '_soc_graph_sentim_raw' + output_extension)
    print_to_csv(make_matrix_graph(naive_quoted_social_graph_sentim_normalized, authors),
                 filename + '/' + output_name + '_naive_quoted' + '_soc_graph_sentim_normalized' + output_extension)

    return reply_count_naive_quoted


def original_quoting(authors, authors_index, authors_pattern, data, filename):
    # noinspection PyUnusedLocal
    social_graph_clean_raw = [[0 for x in range(len(authors))] for x in range(len(authors))]
    # noinspection PyUnusedLocal
    social_graph_clean_normalized = [[0 for x in range(len(authors))] for x in range(len(authors))]
    # noinspection PyUnusedLocal
    social_graph_sentim_raw = [[0 for x in range(len(authors))] for x in range(len(authors))]
    # noinspection PyUnusedLocal
    social_graph_sentim_normalized = [[0 for x in range(len(authors))] for x in range(len(authors))]

    print('    Calculate original quoting')

    reply_count = get_reply_count(data, authors_pattern,
                                  social_graph_clean_raw,
                                  social_graph_clean_normalized,
                                  social_graph_sentim_raw,
                                  social_graph_sentim_normalized,
                                  authors_index)

    print_to_csv(make_matrix_graph(social_graph_clean_raw, authors),
                 filename + '/' + output_name + '_original_quoted' + '_soc_graph_clean_raw' + output_extension)
    print_to_csv(make_matrix_graph(social_graph_clean_normalized, authors),
                 filename + '/' + output_name + '_original_quoted' + '_soc_graph_clean_normalized' + output_extension)
    print_to_csv(make_matrix_graph(social_graph_sentim_raw, authors),
                 filename + '/' + output_name + '_original_quoted' + '_soc_graph_sentim_raw' + output_extension)
    print_to_csv(make_matrix_graph(social_graph_sentim_normalized, authors),
                 filename + '/' + output_name + '_original_quoted' + '_soc_graph_sentim_normalized' + output_extension)

    return reply_count


def print_to_csv(data, file=output_file):
    with open(file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        for row in data:
            csvwriter.writerow(list(row))


def read_csv(file=input_file):
    data = []
    with open(file, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        for row in csv_reader:
            data.append(row)

    return data


def get_authors(data):
    authors = set()

    for row in data:
        authors.add(row[1])

    return authors


def get_authors_list(data):
    authors = []

    for row in data:
        authors.append(row[1])

    return authors


def main():
    key_files = [f for f in listdir('.') if isfile(join('.', f)) and f.startswith('keywords_') and f.endswith('.tsv')]

    if not key_files:
        raise ValueError('No one input file found!')

    for keywords_file in key_files:
        print('Calculating post statistic for ' + keywords_file)

        data = read_csv(keywords_file)
        count = len(data) // 10
        delimiter = float(data[count][1])
        keywords = [x[0] for x in data if float(x[1]) >= delimiter]

        data = read_csv()

        filename, file_extension = os.path.splitext(keywords_file)

        if filename not in listdir('.'):
            os.mkdir(filename)

        authors = list(sorted(get_authors(data)))
        authors_index = {x: i for i, x in enumerate(authors)}
        authors_pattern = '|'.join(authors).replace('.', '\.')

        words_count = get_words_count(data)
        links_count = get_links_count(data)

        reply_count = original_quoting(authors, authors_index, authors_pattern, data, filename)
        reply_count_naive_quoted = naive_quoting(authors, authors_index, data, filename)

        positions = get_positions(data)
        sentiment = get_sentiment(data)
        keywords_count = get_keywords_count(data, keywords)
        utility = get_utility(data)

        post_statistics = get_post_stat(keywords_count, links_count, positions, reply_count, sentiment,
                                        utility, words_count, get_authors_list(data))

        post_statistics_naive_quoted = get_post_stat(keywords_count, links_count, positions,
                                                     reply_count_naive_quoted, sentiment,
                                                     utility, words_count, get_authors_list(data))

        num_of_threads = get_threads_count(data, authors, authors_index)

        print_to_csv(post_statistics, filename + '/' +
                     'stat_' + output_name + '_original_quoted' + output_extension)
        print_to_csv(post_statistics_naive_quoted, filename + '/' +
                     'stat_' + output_name + '_naive_quoted' + output_extension)
        print_to_csv(num_of_threads, filename + '/' + output_name + '_threads_count' + output_extension)


if __name__ == '__main__':
    main()
