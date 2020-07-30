# supress future warning
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
# supress deprecation
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import tensorflow as tf
import numpy as np
import joblib
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser

# argument
parser = ArgumentParser()
parser.add_argument('--unit', default='subword', type=str)
parser.add_argument('--feat_dir', default='data/features_mfcc_bpe_5k/', type=str)
parser.add_argument('--save_dir', default='data/tfrecord_bpe_5k/', type=str)
args = parser.parse_args()



# setup
UNIT = args.unit
MAXLEN = 1710


def load_feats(path, cat):
    partitions = np.sort(glob(path+"/"+cat+"_feats*"))
    feats = []
    for p in partitions:
        feats_ = joblib.load(p)
        feats = np.append(feats, feats_)   
    return feats

# create tfrecords for dataset 

def create_tfrecords(X, y, filename, num_files=5):
    ''' Create tfrecords for dataset 
    Args 
        X: Input acoustic features.
        y: Input token id sequence.
    '''
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    feats = X
    tokens = y

    # Check if the number of sample points matches.
    assert len(feats) == len(tokens)

    num_records_per_file = len(feats) // num_files
    
    total_count = 0  
    
    print("create training dataset....")
    for i in range(num_files):
        # tfrecord writer: write record into files
        count = 0
        writer = tf.python_io.TFRecordWriter(filename+ '-' + str(i+1) + '.tfrecord')

        # put remaining records in last file
        st = i * num_records_per_file                                              # start point (inclusive)
        ed = (i+1) * num_records_per_file if i != num_files-1 else len(tokens)     # end point (exclusive)

        for feat, token in tqdm(zip(feats[st : ed], tokens[st : ed])):
            
            # Construct 'example' object containing 'feat', 'shape', 'token':
            #  {
            #   'feat':  tensor, flattened mfcc or fbank,
            #   'shape': tensor, input feature shape,
            #   'token': tensor, a list of word id which describes input acoustic feature.
            #  }
            example = tf.train.Example(features=tf.train.Features(feature={
                'feat': _float_feature(feat.flatten()),
                'shape': _int64_feature(feat.shape),
                'token': _int64_feature(token),
            }))

            count += 1
            writer.write(example.SerializeToString())

        print("create {}-{}.tfrecord -- contains {} records".format(filename, str(i+1), count))
        total_count += count
        writer.close()
    print("Total records: {}".format(total_count))



# 100h, 360h, 500h
feat_dir = args.feat_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for h in [100, 360, 500]:
    prefix = "train-{}".format(h)
    train_feats = load_feats(feat_dir, prefix)
    train_featlen = np.load(
            feat_dir + "/{}-featlen.npy".format(prefix), allow_pickle=True)
    train_tokens = np.load(
            feat_dir + "/{}_{}s.npy".format(prefix, UNIT), allow_pickle=True)

    assert len(train_feats) == len(train_tokens)

    X = train_feats[train_featlen < MAXLEN]
    y = train_tokens[train_featlen < MAXLEN]

    create_tfrecords(X, y, args.save_dir+predfix, h // 100)

# dev, test
for prefix in ["dev", "test"]:
    eval_feats = load_feats(feat_dir, prefix)
    eval_featlen = np.load(
            feat_dir + "/{}-featlen.npy".format(prefix), allow_pickle=True)
    eval_tokens = np.load(
            feat_dir + "/{}_{}s.npy".format(prefix, UNIT), allow_pickle=True)

    assert len(eval_feats) == len(eval_tokens)

    create_tfrecords(X, y, args.save_dir+predfix, 1)
