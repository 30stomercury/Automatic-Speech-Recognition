# supress future warning
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# supress deprecation
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import os
import sys
import logging
import json
from glob import glob
import joblib
import numpy as np
import tensorflow as tf
from las.utils import convert_idx_to_string, get_save_vars
from las.las import Listener, Speller, LAS
from utils.tokenizer import SubwordEncoder, CharEncoder
from tfrecord_data_loader import tfrecord_iterator, data_parser, get_num_records
from las.arguments import parse_args

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # set your device id
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# arguments
args = parse_args()

# set log
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)s:%(message)s', 
                    level=logging.INFO,
                    datefmt='%I:%M:%S')

print('=' * 60 + '\n')
logging.info('Parameters are:\n%s\n', json.dumps(vars(args), sort_keys=False, indent=4))
print('=' * 60 + '\n')

# init session 
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# tfrecord
training_filenames = [
    "data/tfrecord_bpe_5k/train-100-1.tfrecord", "data/tfrecord_bpe_5k/train-360-1.tfrecord",
    "data/tfrecord_bpe_5k/train-360-2.tfrecord", "data/tfrecord_bpe_5k/train-360-3.tfrecord"
]

# load from previous output
try:
    print("Load features...")
    train_iter, types, shapes = tfrecord_iterator(training_filenames, data_parser)
    num_train_records = get_num_records(training_filenames)
    print('Number of train records in training files: {}'.format(
        num_train_records))

# process features
except:
    raise Exception("Run preprocess.py, create_tfrecord.py first")



# tokenize tools: Using subword unit.
tokenizer = SubwordEncoder(args.subword_dir)
args.vocab_size = tokenizer.get_vocab_size()
id_to_token = tokenizer.id_to_token


# init model 
las =  LAS(args, Listener, Speller, id_to_token)


# build batch iterator
logging.info("Build batch iterator...")
train_xs, train_ys = train_iter.get_next()

# build train graph 
logging.info("Build train graph (please wait)...")
loss, train_op, global_step, train_logits, alphas, train_summary, sample_rate = las.train(train_xs, train_ys)


# save model
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

var_list = get_save_vars()
saver = tf.train.Saver(var_list=var_list, max_to_keep=30)
ckpt = tf.train.latest_checkpoint(args.save_dir)


if ckpt is None:
    sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, ckpt)


# init iterator and graph
sess.run(train_iter.initializer)

# summary
summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

# info
print('=' * 60 + '\n')
logging.info("Training command: python3 {}".format(" ".join(sys.argv)))
print('=' * 60 + '\n')
logging.info("Total weights: {}".format(
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

# training
num_train_batches = num_train_records // 48 + int(num_train_records % 32 != 0)
training_steps = num_train_batches * args.epoch
logging.info("Total num train batches: {}".format(num_train_batches))
logging.info("Training...")
loss_ = []

for step in range(training_steps):
    batch_loss, gs, _, logits, train_gt, tfrate = sess.run(
                        [loss, global_step, 
                            train_op, train_logits, train_ys, sample_rate])

    if args.verbose > 0:
        logging.info("HYP: {}".format(
            convert_idx_to_string(np.argmax(logits, -1)[0], id_to_token, args.unit)))

        logging.info("REF: {}\n".format(
            convert_idx_to_string(train_gt[0][0], id_to_token, args.unit)))

    logging.info("Step: {}, Loss: {:.3f}, tf rate: {:.3f}".format(gs, batch_loss, tfrate))
    loss_.append(batch_loss)
    if gs and gs % num_train_batches == 0:
        ave_loss = np.mean(loss_)
        e_ =  gs // num_train_batches
        logging.info('=' * 19 + ' Epoch %d, Step %d, Ave loss %f' + '=' * 19 + '\n', e_, gs, ave_loss)
        saver.save(sess, args.save_dir+"/las_E{}".format(e_))      
        loss_ = []  

