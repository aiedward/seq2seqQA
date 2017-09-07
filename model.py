import tensorflow as tf
import numpy as np
import data_utils
from tensorflow.contrib import layers


ENCODER_UNITS = 10
DECODER_UNITS = 10
EMBEDDING_SIZE = 10


sess = tf.Session()
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=ENCODER_UNITS)
questions, questions_seq_length, answers, max_question_length = data_utils.get_data('questions_tokenized.txt',
                                                                                    'answers_tokenized.txt')
inputs = tf.placeholder(tf.int32, [None, None])
decoder_inputs = tf.placeholder(tf.int32, [None, None])

vocabulary = data_utils.initialize_vocabulary('vocabulary.txt')
embeddings = tf.Variable(tf.random_uniform([len(vocabulary), EMBEDDING_SIZE], -1.0, 1.0), dtype=tf.float32)

inputs_embed = tf.nn.embedding_lookup(embeddings, inputs)
decoder_inputs_embed = tf.nn.embedding_lookup(embeddings, decoder_inputs)

questions_seq_length_pc = tf.placeholder(tf.int32, [None])
answers_seq_length_pc = tf.placeholder(tf.int32, [None])

encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, inputs_embed, dtype=tf.float32,
                                                  sequence_length=questions_seq_length_pc)

decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=DECODER_UNITS)

decoder_output, decoder_state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embed, dtype=tf.float32,
                                                  sequence_length=answers_seq_length_pc, initial_state=encoder_state)

# TODO: Projection layer
# TODO: Loss function

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess.run(init_op)

encoder_output, encoder_state = sess.run([encoder_output, encoder_state], feed_dict={inputs: questions,
                                     questions_seq_length_pc: np.array(questions_seq_length)})
print()
