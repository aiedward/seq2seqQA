import os
import tensorflow as tf
import nltk
from tqdm import *
import argparse
import numpy as np


FLAGS = None


def create_dictionary_cmd():
    return create_dictionary(FLAGS.path)


def create_dictionary(documents_path):
    dictionary = {}
    rev_dictionary = []
    stats = {}
    i = 0
    for doc in [doc for doc in os.listdir(documents_path) if 'txt' in doc]:
        print('processing ' + doc)
        with tf.gfile.GFile(documents_path + '/' + doc, mode='r') as doc_file:
            lines = doc_file.readlines()
            for line in tqdm(lines):
                tokens = nltk.word_tokenize(line)
                for token in tokens:
                    if token not in dictionary:
                        dictionary[token] = i
                        rev_dictionary.append(token)
                        i+=1
                    count = stats.get(token, 0)
                    count += 1
                    stats[token] = count
    np.save(documents_path + '/stats.npy', stats)
    with tf.gfile.GFile(documents_path + '/dictionary.txt', mode='w') as dict_file:
        for word in rev_dictionary:
            dict_file.write(word + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_dic = subparsers.add_parser('dict', help='create dictionary')
    parser_dic.set_defaults(func=create_dictionary_cmd)
    parser_dic.add_argument('-path', type=str, required=True, help='Dictionary path')

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.func:
        FLAGS.func()