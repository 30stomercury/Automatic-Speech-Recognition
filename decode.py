# supress future  warnings
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# supress deprecation
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import json
import os
import sys
import logging
import joblib
import numpy as np
import tensorflow as tf
from las.beam_search import BeamSearch
from utils.tokenizer import SubwordEncoder, CharEncoder
from las.utils import convert_idx_to_string, wer, edit_distance
from las.arguments import parse_args
from las.las import Listener, Speller, LAS  # load las
from data_loader import batch_gen
from train_lm import load_vocab           
from lang.char_rnn_model import *           # load language model

os.environ['CUDA_VISIBLE_DEVICES'] = '2'    # set your decive number
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

_DECODING_THRESHOLD = 160 # The threshold for long utterance. If > this threshold, split it into two sub utterances for better performance.

def load_lm(init_dir, model_path):
    with open(os.path.join(init_dir, 'result.json'), 'r') as f:
        result = json.load(f)
    params = result['params']
    vocab_file = os.path.join(init_dir, 'vocab.json')
    vocab_index_dict, index_vocab_dict, vocab_size = load_vocab(vocab_file, 'utf-8')
    # Create graphs
    logging.info('Creating rnnlm graph')
    with tf.name_scope('evaluation'):
        params["num_unrollings"] = 1
        params["batch_size"] = 1
        lm = CharRNN(is_training=False, use_batch=True, **params)
    return lm

def restore_lm(sess, save_path):
    with tf.name_scope('evaluation'):
        var_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '')
        var_list = [i[0] for i in tf.train.list_variables(save_path)]
        var_lm = {}
        for v in var_all:
            if v.name.split(":")[0] in var_list:
                var_lm[v.name.split(":")[0]] = v

        #logging.info("Restore language model.")
        # create restore dict for decode scope
        saver_lm = tf.train.Saver(name='checkpoint_saver', var_list=var_lm)
        saver_lm.restore(sess, save_path)
        logging.info("Rnnlm restored: {}".format(save_path))

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

if args.apply_lm:
    logging.info("Apply RNNLM...")
    # Prepare parameters.
    init_dir = "lang/output/"
    model_path = "lang/output/best_model/model-45436"
    lm = load_lm(init_dir, model_path)
else:
    lm = None

sess = tf.Session()

# load from previous output
try:
    logging.info("Load features...")
    dev_feats = joblib.load(args.feat_dir+"/{}_feats.pkl".format(args.split))
    dev_featlen = np.load(args.feat_dir+"/{}_featlen.npy".format(args.split), allow_pickle=True)
    dev_tokens = np.load(args.feat_dir+"/{}_{}s.npy".format(args.split, args.unit), allow_pickle=True)
    dev_tokenlen = np.load(args.feat_dir+"/{}_{}len.npy".format(args.split, args.unit), allow_pickle=True)

# process features
except:
    raise Exception("Run preprocess.py first")


# tokenize tools: Using subword unit.
tokenizer = SubwordEncoder(args.subword_dir)
args.vocab_size = tokenizer.get_vocab_size()
id_to_token = tokenizer.id_to_token


# init model 
las =  LAS(args, Listener, Speller, token_to_id)

# build search decoder
bs = BeamSearch(args, las, token_to_id, lm)

# restore
ckpt = bs.restore_las(sess, args.save_dir, args.restore_epoch)
logging.info("LAS restored: {}".format(ckpt))

if args.apply_lm:
    restore_lm(sess, model_path)
    logging.info("RNNLM restored: {}".format(model_path))

# sort by length
sorted_id = np.argsort(dev_tokenlen)
dev_feats, dev_featlen, dev_tokens = \
            dev_feats[sorted_id], dev_featlen[sorted_id], dev_tokens[sorted_id]

error = 0
N = 0
count = 0
total_utt = len(dev_feats)
logging.info("Decoding...")
for audio, audiolen, y in zip(dev_feats, dev_featlen, dev_tokens):
    if args.convert_rate*audiolen > _DECODING_THRESHOLD:
        # split into sub utterances for decoding
        l = int(args.convert_rate*audiolen) // 100
        m = audiolen // l
        print(l, m)
        hyp = ""
        for i in range(l):
            xs = (np.expand_dims(audio[i*m:(i+1)*m], 0), np.expand_dims(m, 0))
            beam_states = bs.decode(sess, xs)
            hyp += convert_idx_to_string(beam_states[-1].token_ids[1:], id_to_token, args.unit)
    else:
        xs = (np.expand_dims(audio, 0), np.expand_dims(audiolen, 0))
        beam_states = bs.decode(sess, xs)
        hyp = convert_idx_to_string(beam_states[-1].token_ids[1:], id_to_token, args.unit)

    ref = convert_idx_to_string(y, id_to_token, args.unit)
    dist, n  = edit_distance(ref.split(" "), hyp.split(" "))
    error += dist
    N += n
    logging.info("Utt {}/{}, WER: {}".format(count, total_utt, dist/n))
    count += 1
    if args.verbose > 0:
        logging.info("REF | {}".format(ref))
        logging.info("HYP | {}\n".format(hyp))

logging.info("Dev WER: {}".format(error/N))
