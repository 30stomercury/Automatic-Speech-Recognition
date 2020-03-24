# supress future  warnings
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# supress deprecation
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import os
import sys
import logging
import json
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from las.utils import convert_idx_to_string, wer
from las.arguments import parse_args
from las.las import Listener, Speller, LAS
from data_loader import batch_gen
from utils.text import text_encoder


os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # set your device id
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# arguments
args = parse_args()


# set logging
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)s:%(message)s', 
                    level=logging.INFO,
                    datefmt='%I:%M:%S')

print('=' * 60 + '\n')
logging.info('Parameters are:\n%s\n', json.dumps(vars(args), sort_keys=False, indent=4))
print('=' * 60 )


# init session 
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# load from previous output
try:
    logging.info("Load features...")
    dev_feats = np.load(args.feat_path+"/dev_feats.npy", allow_pickle=True)
    dev_featlen = np.load(args.feat_path+"/dev_featlen.npy", allow_pickle=True)
    dev_tokens = np.load(args.feat_path+"/dev_chars.npy", allow_pickle=True)
    dev_tokenlen = np.load(args.feat_path+"/dev_charlen.npy", allow_pickle=True)
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']

# process features
except:
    raise Exception("Run preprocess.py first")

# tokenizer 
tokenizer = text_encoder(args.unit, special_tokens)
id_to_token = tokenizer.id_to_token
args.vocab_size = tokenizer.get_vocab_size()

# init model 
las =  LAS(args, Listener, Speller, id_to_token)

# build batch iterator
logging.info("Build batch iterator...")
dev_iter, num_dev_batches = batch_gen(
            dev_feats, dev_tokens, dev_featlen, dev_tokenlen, args.batch_size, args.feat_dim, True, is_training=False)
dev_xs, dev_ys = dev_iter.get_next()

# build train, inference graph 
logging.info("Build train, inference graph (please wait)...")
dev_logits, y_hat = las.inference(dev_xs)

# saver
saver = tf.train.Saver(max_to_keep=100)
ckpt = tf.train.latest_checkpoint(args.save_path)

if args.restore_epoch != -1:
    ckpt = args.save_path+"/las_E{}".format(args.restore_epoch)

saver.restore(sess, ckpt)

# init iterator and graph
sess.run(dev_iter.initializer)

# info
print('=' * 60)
logging.info("Training command: python3 {}".format(" ".join(sys.argv)))
print('=' * 60)
logging.info("Total weights: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

output_id = []
gt_id = []
texts_pred = []
texts_gt = []

# collect hypothesis
for _ in tqdm(range(num_dev_batches)):
    pred, gt = sess.run([y_hat, dev_ys])
    output_id += pred.tolist()
    gt_id += gt[0].tolist()

# conver into chars
for i in range(len(output_id)):
    hyp_ =  convert_idx_to_string(output_id[i], id_to_token, args.unit)
    gt_ =  convert_idx_to_string(gt_id[i], id_to_token, args.unit)
    texts_pred.append(hyp_)
    texts_gt.append(gt_)

with open(args.log_path+"/test_pred.txt", 'w') as fout:
    fout.write("\n".join(texts_pred))

with open(args.log_path+"/test_gt.txt", 'w') as fout:
    fout.write("\n".join(texts_gt))

# evaluate WER
res = []
for i in range(len(texts_gt)):
    ref = texts_gt[i]
    hyp = texts_pred[i]
    res.append(wer(ref.split(" "), hyp.split(" ")))

logging.info("WER: {}".format(np.mean(res)))
