import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import tensorflow as tf
import numpy as np
import joblib
from glob import glob
from tqdm import tqdm

def load_feats(path, cat):
    partitions = np.sort(glob(path+"/"+cat+"_feats*"))
    feats = []
    for p in partitions:
        feats_ = joblib.load(p)
        feats = np.append(feats, feats_)   
    return feats

# create tfrecords for dataset 

def create_tfrecords(X, y, filename, num_files=5):
    ''' create tfrecords for dataset '''
    
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

        for feat, token in tqdm(zip(feats, tokens)):
            
            # construct 'example' object containing 'feat', 'token' 
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


unit = 'subword'
feat_dir = 'data/LibriSpeech-360/features_mfcc/'
#train_feats_100 = joblib.load(feat_dir+"/train_feats.pkl")
train_feats_100 = load_feats(feat_dir, "train")
train_tokens_100 = np.load(
    feat_dir+"/train_{}s.npy".format(unit), allow_pickle=True)
train_featlen = np.load(
    feat_dir+"/train_featlen.npy", allow_pickle=True)
train_tokens = np.load(
    feat_dir+"/train_{}s.npy".format(unit), allow_pickle=True)
train_tokenlen = np.load(
    feat_dir+"/train_{}len.npy".format(unit), allow_pickle=True)

X = train_feats_100
y = train_tokens_100
create_tfrecords(X, y, 'data/tfrecord/train-360', 10)
