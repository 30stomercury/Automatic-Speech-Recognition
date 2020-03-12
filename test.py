import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from las.utils import convert_idx_to_string
from las.arguments import parse_args
from las.las import Listener, Speller, LAS
from data_loader import batch_gen
from utils.text import text_encoder

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# arguments
args = parse_args()

# init session 
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# load from previous output
try:
    print("Load features...")
    train_feats = np.load(args.feat_path+"/train_feats.npy", allow_pickle=True)
    train_featlen = np.load(args.feat_path+"/train_featlen.npy", allow_pickle=True)
    train_tokens = np.load(args.feat_path+"/train_chars.npy", allow_pickle=True)
    train_tokenlen = np.load(args.feat_path+"/train_charlen.npy", allow_pickle=True)
    dev_feats = np.load(args.feat_path+"/dev_feats.npy", allow_pickle=True)
    dev_featlen = np.load(args.feat_path+"/dev_featlen.npy", allow_pickle=True)
    dev_tokens = np.load(args.feat_path+"/dev_chars.npy", allow_pickle=True)
    dev_tokenlen = np.load(args.feat_path+"/dev_charlen.npy", allow_pickle=True)
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']

# process features
except:
    raise Exception("Run preprocess.py first")

# tokenizer 
tokenizer = text_encoder(args.unit, special_tokens, args.corpus_path)
id_to_token = tokenizer.id_to_token
args.vocab_size = tokenizer.get_vocab_size()

# init model 
las =  LAS(args, Listener, Speller, id_to_token)

# build batch iterator
print("Build batch iterator...")
train_iter, num_train_batches = batch_gen(
            train_feats, train_tokens, train_featlen, train_tokenlen, args.batch_size, args.feat_dim, args.bucketing, is_training=True)
dev_iter, num_dev_batches = batch_gen(
            dev_feats, dev_tokens, dev_featlen, dev_tokenlen, args.batch_size, args.feat_dim, True, is_training=False)
train_xs, train_ys = train_iter.get_next()
dev_xs, dev_ys = dev_iter.get_next()

# build train, inference graph 
print("Build train, inference graph (please wait)...")
dev_logits, y_hat = las.inference(dev_xs)

# saver
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
if not os.path.exists(args.corpus_path):
    os.makedirs(args.corpus_path)

saver = tf.train.Saver(max_to_keep=100)

ckpt = tf.train.latest_checkpoint(args.save_path)
ckpt = "model/las_v11/las_E{}".format(args.restore)
saver.restore(sess, ckpt)

# init iterator and graph
sess.run(train_iter.initializer)
sess.run(dev_iter.initializer)

# info
print("INFO: Training command:"," ".join(sys.argv))
print("INFO: Total weights:",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
#print(tf.trainable_variables())
print("-----------ckpt: {}-----------".format(ckpt))

output_id = []
gt_id = []
texts_pred = []
texts_gt = []

for _ in tqdm(range(num_dev_batches)):
    pred, gt = sess.run([y_hat, dev_ys])
    output_id += pred.tolist()
    gt_id += gt[0].tolist()

for i in range(len(output_id)):
    hyp_ =  convert_idx_to_string(output_id[i], id_to_token, args.unit)
    gt_ =  convert_idx_to_string(gt_id[i], id_to_token, args.unit)
    texts_pred.append(hyp_)
    texts_gt.append(gt_)

with open(args.corpus_path+"/test_pred.txt", 'w') as fout:
    fout.write("\n".join(texts_pred))
with open(args.corpus_path+"/test_gt.txt", 'w') as fout:
    fout.write("\n".join(texts_gt))

def wer(s1,s2):

    d = np.zeros([len(s1)+1,len(s2)+1])
    d[:,0] = np.arange(len(s1)+1)
    d[0,:] = np.arange(len(s2)+1)

    for j in range(1,len(s2)+1):
        for i in range(1,len(s1)+1):
            if s1[i-1] == s2[j-1]:
                d[i,j] = d[i-1,j-1]
            else:
                d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+1)

    return d[-1,-1]/len(s1)

res = []
for i in range(len(texts_gt)):
    ref = texts_gt[i]
    hyp = texts_pred[i]

    res.append(wer(ref.split(" "), hyp.split(" ")))
print("#"*50)
print("WER:", np.mean(res))
