import tensorflow as tf

flags = tf.app.flags

############################
#    hyper parameters      #
############################

# for data
flags.DEFINE_float('train_percent', 0.7, 'training data training percentage')

# for architecture
flags.DEFINE_integer('seq_length', 5, 'length of sequence data for lstm')
flags.DEFINE_integer('data_dim', 9, 'dimension of input data')
flags.DEFINE_integer('hidden_dim', 10, 'LSTM depth(dim) for learning')
flags.DEFINE_integer('output_dim', 1, 'dimension of output data')

# for training
# flags.DEFINE_integer('batch_size', 128, 'batch size')
# flags.DEFINE_integer('epoch', 50, 'epoch')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_integer('iterations', 10000, 'iterations')

############################
#   environment setting    #
############################
# flags.DEFINE_string('dataset', 'mnist', 'The name of dataset [mnist, fashion-mnist')
# flags.DEFINE_boolean('is_training', True, 'train or predict phase')
# flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')
# flags.DEFINE_string('logdir', 'logdir', 'logs directory')
# flags.DEFINE_integer('train_sum_freq', 100, 'the frequency of saving train summary(step)')
# flags.DEFINE_integer('val_sum_freq', 500, 'the frequency of saving valuation summary(step)')
# flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(epoch)')
# flags.DEFINE_string('results', 'results', 'path for saving results')
flags.DEFINE_string('save_dir', 'saved', 'path for saving results')
flags.DEFINE_boolean('is_loading', False, 'load pre-trained weights')
# Ours

############################
#   distributed setting    #
############################
# flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
# flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
# flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

# Ours

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
