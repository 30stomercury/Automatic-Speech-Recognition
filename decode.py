from las.beam_search import BeamSearch
from las.arguments import *
from las.las import *
from las.utils import *
from data_loader import *
from preprocess import *
import numpy as np
import tensorflow as tf
import os


# arguments
args = parse_args()

# init session 
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# load from previous output
try:
    char2id, id2char = lookup_dicts(special_chars)
    print("Load features...")
    dev_feats = np.load(args.feat_path+"/dev_feats.npy", allow_pickle=True)
    dev_featlen = np.load(args.feat_path+"/dev_featlen.npy", allow_pickle=True)
    dev_chars = np.load(args.feat_path+"/dev_chars.npy", allow_pickle=True)
    dev_charlen = np.load(args.feat_path+"/dev_charlen.npy", allow_pickle=True)
    special_chars = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
    char2id, id2char = lookup_dicts(special_chars)

# process features
except:
    raise Exception("Run preprocess.py first")

# init model 
args.vocab_size = len(char2id)
las =  LAS(args, Listener, Speller, char2id, id2char)

# build search decoder
bs = BeamSearch(args, las, char2id)

# create restore dict for decode scope
var = {}
var_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '')
var_att = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'decode/attention/Variable')
var_decode = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'decode/')

for v in var_all:
    if v in var_att:
        var['Speller/while/' + v.name.split(":")[0]] = v
    elif v in var_decode and v not in var_att:
        var['Speller/' + v.name.split(":")[0]] = v
    else:
        var[v.name.split(":")[0]] = v

# restore
saver = tf.train.Saver(var_list=var)
ckpt = tf.train.latest_checkpoint(args.save_path)

"""
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file(ckpt, all_tensors=True, tensor_name='')
"""

if ckpt is None:
    sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, ckpt)

for audio, audiolen, y in zip(dev_feats, dev_featlen, dev_chars):
    xs = (np.expand_dims(audio, 0), np.expand_dims(audiolen, 0))
    beam_states = bs.decode(sess, xs)
    for i in range(len(beam_states)):
        print("Hyposis_{}|".format(i), convert_idx_to_string(beam_states[i].char_ids[1:], id2char))
    print("Ground    |", convert_idx_to_string(y, id2char))
    print("\n")
