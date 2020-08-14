# supress future warning
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# supress deprecation
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import numpy as np
import os



# get the number of records in training files
def get_num_records(files):
  count = 0
  for fn in files:
    for record in tf.python_io.tf_record_iterator(fn):
      count += 1
  return count


def data_parser(record):
  ''' Parse record from .tfrecord file and create training record
    
  Args 
      record: Record extracted from .tfrecord.
  '''

  keys_to_features = {
      'feat': tf.VarLenFeature(dtype=tf.float32),
      'shape': tf.FixedLenFeature([3], dtype=tf.int64),
      'token': tf.VarLenFeature(dtype=tf.int64)
  }

  # features contains - 'feats', 'tokens'
  features = tf.parse_single_example(record, features=keys_to_features)

  feat = features['feat']  # sparse tensor
  feat = tf.sparse_tensor_to_dense(feat)
  shape = tf.cast(features['shape'], tf.int32)
  feat = tf.reshape(feat, [shape[0], shape[1], 3])
  featlen = shape[0]

  token = features[
      'token'].values
  token = tf.cast(token, tf.int32)
  tokenlen = tf.shape(token)[0]

  return (feat, featlen), (token, tokenlen)

def tfrecord_iterator(filenames, record_parser, feat_dim=13, is_training=True):
  ''' Create iterator to eat tfrecord dataset 

    Args
        filenames:     list, a list of filenames(string).
        record_parser: function, a parser that read tfrecord and create example record.

    Return 
        iterator:      An Iterator providing a way to 
                       extract elements from the created dataset.
        output_types:  The output types of the created dataset.
        output_shapes: The output shapes of the created dataset.
  '''

  def _element_length_fn(xs, ys):
    feats = xs[0]
    tokens = ys[0]
    return tf.shape(feats)[0]

  if is_training:
    # max train_featlen: 2971, max train_tokenlen: 219
    bucket_upper_bound = [639, 1062, 1275, 1377, 1449, 1506, 1563, 1710]
    max_tokenlen = 219
  else:
    # max dev_featlen: 3262, max train_tokenlen: 138
    # max test_featlen: 3493, max test_tokenlen: 227
    bucket_upper_bound = [639, 1062, 1275, 1377, 1449, 1506, 1563, 3600]
    max_tokenlen = 227

  bucket_batch_limit = [96, 48, 48, 48, 48, 48, 48, 48, 48]

  # TODO move 13, 3 to parameters
  shapes = shapes = (([None, feat_dim, 3], []), ([max_tokenlen], []))
  files = tf.data.Dataset.list_files(filenames, shuffle=True)
  dataset = files.interleave(map_func=tf.data.TFRecordDataset, cycle_length=10)
  dataset = dataset.map(record_parser, num_parallel_calls=16)
  dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(element_length_func=_element_length_fn, 
                                                                       bucket_boundaries=bucket_upper_bound,
                                                                       bucket_batch_sizes=bucket_batch_limit,
                                                                       padded_shapes=shapes,
                                                                       pad_to_bucket_boundary=True))
  dataset.prefetch(1)

  if is_training:
    dataset = dataset.shuffle(64)        # shuffle the dataset
    dataset = dataset.repeat()           # repeat dataset infinitely

  iterator = dataset.make_initializable_iterator()
  output_types = dataset.output_types
  output_shapes = dataset.output_shapes

  return iterator, output_types, output_shapes

if __name__ == '__main__':
    training_filenames = "data/tfrecord_fbank_bpe_5k/train-360*.tfrecord"
    train_iter, types, shapes = tfrecord_iterator(training_filenames, data_parser, 80)
    print(types, shapes)

    sess = tf.Session()
    # init iterator and graph
    sess.run(train_iter.initializer)

    train_xs, train_ys = train_iter.get_next()
    step = tf.reduce_max(train_ys[1])
    xs, ys, step_ = sess.run([train_xs, train_ys, step])
    print(step_)
    print('xs', xs[0].shape, len(xs[1]))
    print('ys', ys[0].shape, len(ys[1]))
    print(len(ys[1]))
    print(max(ys[1]))
    print(ys[0][0],ys[1])

