import tensorflow as tf
import numpy as np
import data_utils
from tensorflow.contrib import layers
from tqdm import *


ENCODER_UNITS = 10
DECODER_UNITS = 10
EMBEDDING_SIZE = 10


tf.app.flags.DEFINE_boolean('train_mode', True, 'Run in a training mode')
tf.app.flags.DEFINE_integer('num_epochs', 5000, 'Number of epochs')
FLAGS = tf.app.flags.FLAGS

sess = tf.Session()

questions, q_seq_length, answers_inputs, answers_targets, a_seq_length = data_utils.get_data('questions_tokenized.txt',
                                                                                             'answers_tokenized.txt')
inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
decoder_inputs = tf.placeholder(tf.int32, [None, None], name='decoder_inputs')
decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')

vocabulary, rev_vocabulary, vocabulary_size = data_utils.initialize_vocabulary('vocabulary.txt')
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_SIZE], -1.0, 1.0), dtype=tf.float32,
                         name='embeddings')

inputs_embed = tf.nn.embedding_lookup(embeddings, inputs)
decoder_inputs_embed = tf.nn.embedding_lookup(embeddings, decoder_inputs)

questions_seq_length_pc = tf.placeholder(tf.int32, [None], name='questions_sequence_length')
answers_seq_length_pc = tf.placeholder(tf.int32, [None], name='answers_sequence_length')

with tf.variable_scope('encoder'):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=ENCODER_UNITS)
    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, inputs_embed, dtype=tf.float32,
                                                      sequence_length=questions_seq_length_pc)
with tf.variable_scope('decoder'):
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=DECODER_UNITS)
    decoder_output, decoder_state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embed, dtype=tf.float32,
                                                      sequence_length=answers_seq_length_pc,
                                                      initial_state=encoder_state)
    
decoder_logits = layers.fully_connected(decoder_output, vocabulary_size)
decoder_prediction = tf.argmax(decoder_logits, 2)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocabulary_size, dtype=tf.float32),
    logits=decoder_logits)

loss_op = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss_op)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess.run(init_op)

train_writer = tf.summary.FileWriter('/home/thetweak/Developer/agora_qa_seq2seq/log', sess.graph)
tf.summary.scalar('loss', loss_op)
merged_summary = tf.summary.merge_all()

for e in tqdm(range(FLAGS.num_epochs)):
    summary, _ = sess.run([merged_summary, train_op], feed_dict={inputs: questions,
                                                                 questions_seq_length_pc: q_seq_length,
                                                                 decoder_inputs: answers_inputs,
                                                                 decoder_targets: answers_targets,
                                                                 answers_seq_length_pc: a_seq_length})
    train_writer.add_summary(summary, e)
