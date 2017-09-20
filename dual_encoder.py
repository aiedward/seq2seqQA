import tensorflow as tf
import numpy as np
import data_utils
from tensorflow.contrib import layers
from tqdm import *


CONTEXT_ENCODER_UNITS = 10
RESPONSE_ENCODER_UNITS = 10
EMBEDDING_SIZE = 10


tf.app.flags.DEFINE_boolean('train_mode', True, 'Run in a training mode')
tf.app.flags.DEFINE_integer('num_epochs', 5000, 'Number of epochs')
FLAGS = tf.app.flags.FLAGS

sess = tf.Session()

questions, q_seq_length, answers_inputs, answers_targets, a_seq_length = data_utils.get_data('questions_tokenized.txt',
                                                                                             'answers_tokenized.txt')
context_encoder_inputs = tf.placeholder(tf.int32, [None, None], name='context_encoder_inputs')
response_encoder_inputs = tf.placeholder(tf.int32, [None, None], name='response_encoder_inputs')

vocabulary, rev_vocabulary, vocabulary_size = data_utils.initialize_vocabulary('vocabulary.txt')
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_SIZE], -1.0, 1.0), dtype=tf.float32,
                         name='embeddings')

emb_context_encoder_inputs = tf.nn.embedding_lookup(embeddings, context_encoder_inputs)
emb_response_encoder_inputs = tf.nn.embedding_lookup(embeddings, response_encoder_inputs)

questions_seq_length_pc = tf.placeholder(tf.int32, [None], name='questions_sequence_length')
answers_seq_length_pc = tf.placeholder(tf.int32, [None], name='answers_sequence_length')

with tf.variable_scope('context_encoder'):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=CONTEXT_ENCODER_UNITS)
    context_encoder_output, context_encoder_state = tf.nn.dynamic_rnn(cell, emb_context_encoder_inputs,
                                                                      dtype=tf.float32,
                                                                      sequence_length=questions_seq_length_pc)

with tf.variable_scope('response_encoder'):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=RESPONSE_ENCODER_UNITS)
    response_encoder_output, response_encoder_state = tf.nn.dynamic_rnn(cell, emb_response_encoder_inputs,
                                                                        dtype=tf.float32,
                                                                        sequence_length=answers_seq_length_pc)

with tf.variable_scope('learn_matrix'):
    M = tf.get_variable('M', shape=[CONTEXT_ENCODER_UNITS, RESPONSE_ENCODER_UNITS],
                        initializer=tf.truncated_normal_initializer())

generated_context = response_encoder_state.h @ M
dot_product = tf.reduce_sum(tf.multiply(context_encoder_state.h, generated_context), 1, keep_dims=True)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
dot_product_result = sess.run(dot_product, feed_dict={context_encoder_inputs: questions,
                                                      questions_seq_length_pc: q_seq_length,
                                                      response_encoder_inputs: answers_inputs,
                                                      answers_seq_length_pc: a_seq_length})
print(dot_product_result)