import tensorflow as tf
import numpy as np
import data_utils
from tensorflow.contrib import layers
from tqdm import *


CONTEXT_ENCODER_UNITS = 50
RESPONSE_ENCODER_UNITS = 50
EMBEDDING_SIZE = 300


tf.app.flags.DEFINE_boolean('train_mode', True, 'Run in a training mode')
tf.app.flags.DEFINE_integer('num_epochs', 10000, 'Number of epochs')
tf.app.flags.DEFINE_string('model_dir', 'v0', 'Model checkpoint directory')
tf.app.flags.DEFINE_string('questions_train', 'questions_tokenized_emb.txt', 'Tokenized questions for training')
tf.app.flags.DEFINE_string('answers_train', 'answers_tokenized_emb.txt', 'Tokenized answers for training')
tf.app.flags.DEFINE_string('labels', 'labels.txt', 'Labels')
FLAGS = tf.app.flags.FLAGS

sess = tf.Session()

questions, q_seq_length, answers, a_seq_length, labels = data_utils.get_data(FLAGS.questions_train,
                                                                             FLAGS.answers_train,
                                                                             FLAGS.labels)
embeddings_array = np.load('/home/thetweak/Developer/agora_qa_seq2seq/embeddings.npy')

context_encoder_inputs = tf.placeholder(tf.int32, [None, None], name='context_encoder_inputs')
response_encoder_inputs = tf.placeholder(tf.int32, [None, None], name='response_encoder_inputs')
labels_input = tf.placeholder(tf.int32, [None], name='labels')

vocabulary, rev_vocabulary, vocabulary_size = data_utils.initialize_vocabulary('vocabulary_embeddings.txt')
embeddings = tf.Variable(tf.constant(0.0, tf.float32, [vocabulary_size, EMBEDDING_SIZE]), dtype=tf.float32, trainable=False,
                         name='embeddings')

embeddings_placeholder = tf.placeholder(tf.float32, [vocabulary_size, EMBEDDING_SIZE])
embeddings = embeddings.assign(embeddings_placeholder)

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

with tf.variable_scope('bias'):
    b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[CONTEXT_ENCODER_UNITS]))

generated_context = response_encoder_state.h @ M + b
dot_product = tf.squeeze(tf.reduce_sum(tf.multiply(context_encoder_state.h, generated_context), 1, keep_dims=True))
predictions = tf.sigmoid(dot_product)
x_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(labels_input), logits=dot_product))
train_op = tf.train.AdamOptimizer(1e-4).minimize(x_entropy)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
saver = tf.train.Saver()

if FLAGS.train_mode:
    train_writer = tf.summary.FileWriter('log/' + FLAGS.model_dir, sess.graph)
    tf.summary.scalar('x_entropy', x_entropy)
    merged_summary = tf.summary.merge_all()
    top_k_op = tf.nn.top_k(predictions, k=5)
    for e in tqdm(range(FLAGS.num_epochs)):
        summary, _, emb_context_encoder_inputs_r = sess.run([merged_summary, train_op, emb_context_encoder_inputs], feed_dict={context_encoder_inputs: questions,
                                                                                                                               questions_seq_length_pc: q_seq_length,
                                                                                                                               response_encoder_inputs: answers,
                                                                                                                               answers_seq_length_pc: a_seq_length,
                                                                                                                               labels_input: labels,
                                                                                                                               embeddings_placeholder: embeddings_array})
        if e % 100 == 0:
            for i in [6, 2, 15]:
                q_path = 'question_test_{}_tokenized_emb.txt'.format(i)
                question_v, q_v_seq_length, answers_all, a_seq_length_all, _ = data_utils.get_data(q_path,
                                                                                                   'all_answers_tokenized_emb.txt',
                                                                                                   FLAGS.labels)
                top_k, predictions_r = sess.run([top_k_op, predictions], feed_dict={context_encoder_inputs: question_v,
                                                                                    questions_seq_length_pc: q_v_seq_length,
                                                                                    response_encoder_inputs: answers_all,
                                                                                    answers_seq_length_pc: a_seq_length_all,
                                                                                    embeddings_placeholder: embeddings_array})
                correct = i in top_k.indices
                print('Prediction for question {} is {}correct: {}'.format(i, '' if correct else 'not ', top_k.indices))
        train_writer.add_summary(summary, e)
    saver.save(sess, FLAGS.model_dir)
else:
    saver.restore(sess, FLAGS.model_dir)
    top_k_op = tf.nn.top_k(predictions, k=5)
    predictions_r, top_k = sess.run([predictions, top_k_op], feed_dict={context_encoder_inputs: questions,
                                                                        questions_seq_length_pc: q_seq_length,
                                                                        response_encoder_inputs: answers,
                                                                        answers_seq_length_pc: a_seq_length,
                                                                        embeddings_placeholder: embeddings_array})
    print(top_k.indices)