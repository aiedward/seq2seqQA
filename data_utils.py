# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tensorflow.python.platform import gfile
import tensorflow as tf
import re
import argparse
import numpy as np


# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


FLAGS = None


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size=100000,
                      tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.
    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.
    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.
    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].
    Args:
      vocabulary_path: path to the file containing the vocabulary.
    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).
    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab, len(rev_vocab)
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.
    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    Returns:
      a list of integers, the token-ids for the sentence.
    """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.
    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.
    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                                      tokenizer, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def create_voc_command():
    return create_vocabulary(FLAGS.voc, FLAGS.data)


def data_to_tokens_command():
    return data_to_token_ids(FLAGS.data, FLAGS.target, FLAGS.voc)


def get_data(questions_path, answers_path):
    questions = []
    answers = []
    max_question_length = 0
    max_answer_length = 0
    with tf.gfile.GFile(questions_path, mode='r') as questions_file:
        with tf.gfile.GFile(answers_path, mode='r') as answers_file:
            question, answer = questions_file.readline(), answers_file.readline()
            while question and answer:
                question_ids = [int(x) for x in question.split()]
                if len(question_ids) > max_question_length:
                    max_question_length = len(question_ids)
                answer_ids = [int(x) for x in answer.split()]
                if len(answer_ids) > max_answer_length:
                    max_answer_length = len(answer_ids)
                questions.append(np.array(question_ids, dtype=np.int32))
                answers.append(answer_ids)
                question, answer = questions_file.readline(), answers_file.readline()
    questions_result = []
    q_seq_length = []
    for q in questions:
        q_seq_length.append(len(q))
        if len(q) < max_question_length:
            questions_result.append(np.append(q, np.array([0 for _ in range(max_question_length - len(q))],
                                                          dtype=np.int32)))
        else:
            questions_result.append(q)
    answers_inputs = []
    answers_targets = []
    a_seq_length = []
    for a in answers:
        a_seq_length.append(len(a) + 1)  # +1 for EOS token
        a_eos = a + [EOS_ID]
        eos_a = [EOS_ID] + a
        if len(a) < max_answer_length:
            trailing_zeros = [0 for _ in range(max_answer_length - len(a))]  # EOS token
            answers_inputs.append(a_eos + trailing_zeros)
            answers_targets.append(eos_a + trailing_zeros)
        else:
            answers_inputs.append(a_eos)
            answers_targets.append(eos_a)
    return np.array(questions_result), np.array(q_seq_length), np.array(answers_inputs), np.array(answers_targets), np.array(a_seq_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_voc = subparsers.add_parser('vocab', help='create vocabulary')
    parser_voc.set_defaults(func=create_voc_command)
    parser_voc.add_argument('--voc', type=str, required=True, help='Vocabulary path')
    parser_voc.add_argument('--data', type=str, required=True, help='Data path')

    parser_data_to_tok = subparsers.add_parser('data2tok', help='tokenize data')
    parser_data_to_tok.set_defaults(func=data_to_tokens_command)
    parser_data_to_tok.add_argument('--voc', type=str, required=True, help='Vocabulary path')
    parser_data_to_tok.add_argument('--data', type=str, required=True, help='Data path')
    parser_data_to_tok.add_argument('--target', type=str, required=True, help='Target path')

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.func:
        FLAGS.func()