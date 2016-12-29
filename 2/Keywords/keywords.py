import sys

if sys.version_info < (3, 0):
    raise Exception('must use python 3.0 or greater')

import csv
import re
import operator
from os import listdir
from os.path import isfile, join

__author__ = u'Stranger'

stop_words = list({
    'в',
    'как',
    'примерно',
    'общих',
    'всем',
    'на',
    'при',
    'или',
    'это',
    'по',
    'и',
    'от',
    'из',
    'благодаря',
    'затем',
    'все',
    'мы',
    'но',
    'но',
    'не',
    'это',
    'что',
    'так',
    'тот',
    'кто',
    'них',
    'а',
    'чтобы',
    'когда',
    'учитывая',
    'еще',
    'под',
    'ней',
    'собой',
    'понятно',
    'которые',
    'свою',
    'то',
    'она',
    'никак',
    'вообще',
    'ну',
    'очень',
    'то',
    'тут',
    'какая',
    'эта',
    'только',
    'лишь',
    'для',
    'того',
    'тогда',
    'никакого',
    'давайте',
    'вот',
    'все',
    'имхо',
    'хочет',
    'получить',
    'читать',
    'эту',
    'тех',
    'критикуйте',
    'такая',
    'совсем',
    'за',
    'этого',
    'во',
    'время',
    'его',
    'конечно',
    'счет',
    'которое',
    'видно',
    'они',
    'свое',
    'своей',
    'хотел',
    'вижу',
    'слова',
    'работают',
    'нажатии',
    'создает',
    'заполненной',
    'передается',
    'выталкивает',
    'давят',
    'сжимают',
    'создавая',
    'преобразовывается',
    'рассеивается',
    'остановились',
    'написать',
    'возникают',
    'отходят',
    'делать',
    'распространенные',
    'разбирал',
    'знает',
    'сидят',
    'разводящая',
    'сдвинуть',
    'влияет',
    'двигаются',
    'служит',
    'болтались',
    'издавали',
    'гремящий',
    'возвращаются',
    'отпускаем',
    'оказывает',
    'разбираться',
    'появляется',
    'действует',
    'выталкивается',
    'деформируется',
    'возникают',
    'стремятся',
    'падает',
    'позволяет',
    'вернуться',
    'делает',
    'увлекая',
    'разводятся',
    'остается',
    'вернуть',
    'возвращается',
    'теряет',
    'я'})


def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False


def separate_words(text, min_word_return_size):
    splitter = re.compile('[\s]')
    words = []
    for single_word in splitter.split(text):
        single_word = reduce(single_word)
        current_word = single_word.strip().lower()
        # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)
    return words


def split_sentences(text):
    sentence_delimiters = re.compile(u'[.!?]')
    sentences = sentence_delimiters.split(text)
    return sentences


def build_stop_word_regex():
    stop_word_regex_list = []
    for word in stop_words:
        word_regex = r'\b' + word + r'(?![\w-])'  # added look ahead for hyphen
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern


def generate_candidate_keywords(sentence_list, stopword_pattern):
    phrase_list = []
    for s in sentence_list:
        tmp = re.sub(stopword_pattern, '|', s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if phrase != "":
                phrase_list.append(phrase)
    return phrase_list


def reduce(word):
    if len(word) >= 12:
        return word[:7]
        pass
    elif len(word) >= 10:
        return word[:6]
        pass
    elif len(word) >= 6:
        return word[:4]
        pass
    elif len(word) >= 4:
        return word[:3]
        pass
    else:
        return word


def calculate_word_scores(phrase_list):
    word_frequency = {}
    word_degree = {}
    for phrase in phrase_list:
        word_list = separate_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        # if word_list_degree > 3: word_list_degree = 3 #exp.
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree  # orig.
            # word_degree[word] += 1/(word_list_length*1.0) #exp.
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    # Calculate Word scores = deg(w)/frew(w)
    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  # orig.
    # word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
    return word_score


def calculate_word_scores_exp(phrase_list):
    word_frequency = {}
    word_degree = {}
    for phrase in phrase_list:
        word_list = separate_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        # if word_list_degree > 3:
        #     word_list_degree = 3  # exp.
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            # word_degree[word] += word_list_degree  # orig.
            word_degree[word] += 1 / (word_list_length * 1.0)  # exp.
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    # Calculate Word scores = deg(w)/frew(w)
    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        # word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  # orig.
        word_score[item] = word_frequency[item] / (word_degree[item] * 1.0)  # exp.
    return word_score


def calculate_word_frequency(phrase_list):
    word_frequency = {}
    word_degree = {}
    for phrase in phrase_list:
        word_list = separate_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree

    return word_frequency


def generate_candidate_keyword_scores(phrase_list, word_score):
    keyword_candidates = {}
    for phrase in phrase_list:
        keyword_candidates.setdefault(phrase, 0)
        word_list = separate_words(phrase, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_score[word]
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates


def read_csv(input_file):
    input_data = []

    with open(input_file, u'r', newline=u'') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=u'\t')
        for row in csv_reader:
            input_data.append(row)

    return input_data


class Rake:
    def __init__(self):
        self.__stop_words_pattern = build_stop_word_regex()

    def run(self, text):
        sentence_list = split_sentences(text)
        sentence_list = [phrase.lower().strip() for phrase in sentence_list if len(re.findall('\S', phrase))]

        phrase_list = generate_candidate_keywords(sentence_list, self.__stop_words_pattern)
        word_scores = calculate_word_scores(phrase_list)
        word_scores = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)

        return word_scores


class ExpRake:
    def __init__(self):
        self.__stop_words_pattern = build_stop_word_regex()

    def run(self, text):
        sentence_list = split_sentences(text)
        sentence_list = [phrase.lower().strip() for phrase in sentence_list if len(re.findall('\S', phrase))]

        phrase_list = generate_candidate_keywords(sentence_list, self.__stop_words_pattern)
        word_scores = calculate_word_scores_exp(phrase_list)
        word_scores = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)

        return word_scores


class Frequency:
    def __init__(self):
        self.__stop_words_pattern = build_stop_word_regex()

    def run(self, text):
        sentence_list = split_sentences(text)
        sentence_list = [phrase.lower().strip() for phrase in sentence_list if len(re.findall('\S', phrase))]

        phrase_list = generate_candidate_keywords(sentence_list, self.__stop_words_pattern)
        word_scores = calculate_word_frequency(phrase_list)
        word_scores = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)

        return word_scores


def print_to_csv(data, file):
    with open(file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t')
        for row in data:
            csvwriter.writerow(list(row))


def main():
    files = [f for f in listdir('.') if isfile(join('.', f)) and f.endswith('.tsv') and not f.startswith('keywords_')]

    if not files:
        raise ValueError('No one input file found!')

    for file_name in files:
        print('Extracting keywords for ' + file_name)

        data = read_csv(file_name)
        topic_starter = data[0][2]
        topic_starter = re.sub(r'[a-zA-Z0-9:/.,=&#*]{2,}', ' ', topic_starter)  # remove links
        topic_starter = re.sub(r'\(|\)|\-|«|»|Т\.е\.|,|:', ' ', topic_starter)  # remove trash
        topic_starter = re.sub(r'…', '.', topic_starter)  # replace delimiter

        rake = Rake()
        rake_keywords = rake.run(topic_starter)

        exp_rake = ExpRake()
        exp_rake_keywords = exp_rake.run(topic_starter)

        frequency = Frequency()
        freq_keywords = frequency.run(topic_starter)

        print_to_csv(list(rake_keywords), 'keywords_rake.tsv')
        print_to_csv(list(exp_rake_keywords), 'keywords_exp_rake.tsv')
        print_to_csv(list(freq_keywords), 'keywords_frequency.tsv')


if __name__ == u'__main__':
    main()
