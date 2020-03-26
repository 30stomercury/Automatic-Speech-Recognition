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
from utils.text import text_encoder
from las.utils import convert_idx_to_string
from las.arguments import parse_args
from las.las import Listener, Speller, LAS
from data_loader import batch_gen

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # set your device id
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def load_feats(path, cat):
    partitions = np.sort(glob(path+"/"+cat+"_feats*"))
    feats = []
    for p in partitions:
        feats_ = joblib.load(p)
        feats = np.append(feats, feats_)   
    return feats

# load from previous output
try:
    print("Load features...")
    # train
    train_feats = load_feats(args.feat_dir, "train")
    train_featlen = np.load(
        args.feat_dir+"/train_featlen.npy", allow_pickle=True)
    train_tokens = np.load(
        args.feat_dir+"/train_{}s.npy".format(args.unit), allow_pickle=True)
    train_tokenlen = np.load(
        args.feat_dir+"/train_{}len.npy".format(args.unit), allow_pickle=True)
    # aug
    if args.augmentation:
        for factor in [0.9, 1.1]:
            aug_feats = load_feats(args.feat_dir, "speed_{}".format(factor))
            aug_featlen = np.load(
                    args.feat_dir+"/speed_{}_featlen.npy".format(factor), allow_pickle=True)
            train_feats = np.append(train_feats, aug_feats)
            train_featlen = np.append(train_featlen, aug_featlen)
            train_tokens = np.append(train_tokens, train_tokens[:len(aug_feats)])
            train_tokenlen = np.append(train_tokenlen, train_tokenlen[:len(aug_feats)])

# process features
except:
    raise Exception("Run preprocess.py first")

# Limit text length to predefined decoding steps
# train
index = train_tokenlen < args.maxlen
train_feats = train_feats[index]
train_featlen = train_featlen[index]
train_tokens = train_tokens[index]
train_tokenlen = train_tokenlen[index]

# tokenize tools
special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
tokenizer = text_encoder(args.unit, special_tokens)
args.vocab_size = tokenizer.get_vocab_size()
id_to_token = tokenizer.id_to_token

# init model 
las =  LAS(args, Listener, Speller, id_to_token)

# build batch iterator
logging.info("Build batch iterator...")
train_iter, num_train_batches = batch_gen(
                                    train_feats, 
                                    train_tokens, 
                                    train_featlen, 
                                    train_tokenlen, 
                                    args.batch_size, 
                                    args.feat_dim, 
                                    args.bucketing, 
                                    is_training=True)
train_xs, train_ys = train_iter.get_next()

# build train, inference graph 
logging.info("Build train graph (please wait)...")
loss, train_op, global_step, train_logits, alphas, train_summary = las.train(train_xs, train_ys)

# saver
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

saver = tf.train.Saver(max_to_keep=100)

if args.restore_epoch > 0:
    ckpt = args.save_dir + "/las_E{}".format(args.restore_epoch)
else:
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
training_steps = num_train_batches * args.epoch
logging.info("Total num train batches:".format(num_train_batches))
logging.info("Training...")
loss_ = []

for step in range(training_steps):
    batch_loss, gs, _, summary_, logits, train_gt = sess.run(
                        [loss, global_step, train_op, train_summary, train_logits, train_ys])

    if args.verbose > 0:
        logging.info("HYP: {}\nREF: {}".format(
            convert_idx_to_string(np.argmax(logits, -1)[0], id_to_token, args.unit), 
            convert_idx_to_string(train_gt[0][0], id_to_token, args.unit)))

    logging.info("Step: {}, Loss: {}".format(gs, batch_loss))
    summary_writer.add_summary(summary_, gs)
    loss_.append(batch_loss)
    if gs and gs % num_train_batches == 0:
        ave_loss = np.mean(loss_)
        e_ =  gs // num_train_batches
        logging.info('=' * 19 + ' Epoch %d, Step %d, Ave loss %f' + '=' * 19 + '\n', e_, gs, ave_loss)
        saver.save(sess, args.save_dir+"/las_E{}".format(e_))      
        loss_ = []  

summary_writer.close()
