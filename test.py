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
import joblib
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from las.utils import convert_idx_to_string, wer, edit_distance
from las.las import Listener, Speller, LAS
from utils.tokenizer import Subword_Encoder
from las.arguments import parse_args


os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # set your device id
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

# tfrecord
eval_filenames = [
    "data/tfrecord_bpe_5k/dev-1.tfrecord",
]

# load from previous output
try:
    print("Load features...")
    eval_iter, types, shapes = tfrecord_iterator(eval_filenames, training_parser, is_training=False)
    num_eval_records = get_num_records(eval_filenames)
    print('Number of train records in eval files: {}'.format(
        num_eval_records))

# process features
except:
    raise Exception("Run preprocess.py, create_tfrecord.py first")

# tokenize tools: Using subword unit.
tokenizer = Subword_Encoder(args.subword_dir)
args.vocab_size = tokenizer.get_vocab_size()
id_to_token = tokenizer.id_to_token

# init model 
las =  LAS(args, Listener, Speller, id_to_token)

# build batch iterator
logging.info("Build batch iterator...")
eval_xs, eval_ys = eval_iter.get_next()

# build train, inference graph 
logging.info("Build train, inference graph (please wait)...")
eval_logits, y_hat = las.inference(eval_xs)

# saver
saver = tf.train.Saver(max_to_keep=100)
ckpt = tf.train.latest_checkpoint(args.save_dir)

if args.restore_epoch != -1:
    ckpt = args.save_dir+"/las_E{}".format(args.restore_epoch)

saver.restore(sess, ckpt)

# init iterator and graph
sess.run(eval_iter.initializer)

# info
print('=' * 60)
logging.info("Testing command: python3 {}".format(" ".join(sys.argv)))
print('=' * 60)
logging.info("Total weights: {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

output_id = []
gt_id = []
texts_pred = []
texts_gt = []

num_eval_batches = 100

# collect hypothesis
for _ in tqdm(range(num_eval_batches)):
    pred, gt = sess.run([y_hat, eval_ys])
    output_id += pred.tolist()
    gt_id += gt[0].tolist()

# conver into chars
for i in range(len(output_id)):
    hyp_ =  convert_idx_to_string(output_id[i], id_to_token, args.unit)
    gt_ =  convert_idx_to_string(gt_id[i], id_to_token, args.unit)
    texts_pred.append(hyp_)
    texts_gt.append(gt_)

with open(args.log_dir+"/test_pred.txt", 'w') as fout:
    fout.write("\n".join(texts_pred))

with open(args.log_dir+"/test_gt.txt", 'w') as fout:
    fout.write("\n".join(texts_gt))

# evaluate WER
error = 0
N = 0
for i in range(len(texts_gt)):
    ref = texts_gt[i]
    hyp = texts_pred[i]
    e, n = edit_distance(ref.split(" "), hyp.split(" "))
    error += e
    N += n
   
logging.info("WER: {}".format(error/N))
