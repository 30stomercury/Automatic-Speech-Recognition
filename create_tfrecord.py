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
NUM_FILE_PER_TFRECORD = 5000


def load_train_feats(filenames):
    
    feats = []
    for f in filenames:
        print("load", f)
        feats_ = joblib.load(f)
        feats = np.append(feats, feats_)   

    return feats

# create tfrecords for dataset 

def create_tfrecords(X, y, filename, num_files=5, file_start_index=1):
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
        writer = tf.python_io.TFRecordWriter(filename+ '-' + str(i+file_start_index) + '.tfrecord')

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

        print("create {}-{}.tfrecord -- contains {} records".format(filename, str(i+file_start_index), count))
        total_count += count
        writer.close()
    print("Total records: {}".format(total_count))



if __name__ == '__main__':

    # 100h, 360h, 500h
    feat_dir = args.feat_dir

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for h in [100, 360, 500]:
        prefix = "train-{}".format(h)
        train_tokens = np.load(
                feat_dir + "/{}-{}s.npy".format(prefix, UNIT), allow_pickle=True)


        num_files = len(glob(feat_dir+"/"+prefix+"-feats*"))
        num_partitions = h // 50
        filenames = [feat_dir+"/"+prefix+"-feats-{}.pkl".format(i) for i in range(num_files)]

        num_pkl_per_tfrecord = num_files // num_partitions

        st_save_index = 1
        st_token_index = 0
        for i in range(num_partitions):
            
            st = i * num_pkl_per_tfrecord                                                    # start point (inclusive)
            ed = (i+1) * num_pkl_per_tfrecord if i != num_partitions-1 else num_files        # end point (exclusive)
            train_feats = load_train_feats(filenames[st:ed])
            train_tokens_ = train_tokens[st_token_index:st_token_index+len(train_feats)]
            st_token_index += len(train_feats)

            # Shuffle
            rand_idx = np.random.permutation(len(train_tokens_))
            train_feats = train_feats[rand_idx]
            train_featlen = np.array([len(feat) for feat in train_feats])
            train_tokens_ = train_tokens_[rand_idx]

            # Clip to maxlen
            X = train_feats[train_featlen < MAXLEN]
            y = train_tokens_[train_featlen < MAXLEN]
            print(st_save_index, st_token_index)
            create_tfrecords(X, y, args.save_dir+prefix, len(y) // NUM_FILE_PER_TFRECORD, st_save_index)
            st_save_index += len(y) // NUM_FILE_PER_TFRECORD

    # dev, test
    for prefix in ["dev", "test"]:
        eval_feats = joblib.load(feat_dir + "/{}-feats.pkl".format(prefix))
        eval_tokens = np.load(
                feat_dir + "/{}-{}s.npy".format(prefix, UNIT), allow_pickle=True)

        assert len(eval_feats) == len(eval_tokens)

        X = eval_feats
        y = eval_tokens

        create_tfrecords(X, y, args.save_dir+prefix, 1)
