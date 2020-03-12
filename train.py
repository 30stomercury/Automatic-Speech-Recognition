import os
import sys
import numpy as np
import tensorflow as tf
from utils.text import text_encoder
from las.utils import convert_idx_to_string
from las.arguments import parse_args
from las.las import Listener, Speller, LAS
from data_loader import batch_gen

# arguments
args = parse_args()

# init session 
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# load from previous output
try:
    print("Load features...")
    # train
    train_feats = np.load(
        args.feat_path+"/train_feats.npy", allow_pickle=True)
    train_featlen = np.load(
        args.feat_path+"/train_featlen.npy", allow_pickle=True)
    train_tokens = np.load(
        args.feat_path+"/train_{}s.npy".format(args.unit), allow_pickle=True)
    train_tokenlen = np.load(
        args.feat_path+"/train_{}len.npy".format(args.unit), allow_pickle=True)
    # dev
    dev_feats = np.load(
        args.feat_path+"/dev_feats.npy", allow_pickle=True)
    dev_featlen = np.load(
        args.feat_path+"/dev_featlen.npy", allow_pickle=True)
    dev_tokens = np.load(
        args.feat_path+"/dev_{}s.npy".format(args.unit), allow_pickle=True)
    dev_tokenlen = np.load(
        args.feat_path+"/dev_{}len.npy".format(args.unit), allow_pickle=True)
    # aug
    if args.augmentation:
        for factor in [0.9, 1.1]:
            aug_feats = np.load(
                    args.feat_path+"/aug_feats_speed_{}.npy".format(factor), allow_pickle=True)
            aug_featlen = np.load(
                    args.feat_path+"/aug_featlen_speed_{}.npy".format(factor), allow_pickle=True)
            train_feats = np.append(train_feats, aug_feats)
            train_featlen = np.append(train_featlen, aug_featlen)
            train_tokens = np.append(train_tokens, train_tokens[:len(aug_feats)])
            train_tokenlen = np.append(train_tokenlen, train_tokenlen[:len(aug_feats)])

# process features
except:
    raise Exception("Run preprocess.py first")

# Clip text length to predefined decoding steps
# train
index = train_tokenlen < args.maxlen
train_feats = train_feats[index]
train_featlen = train_featlen[index]
train_tokens = train_tokens[index]
train_tokenlen = train_tokenlen[index]

# tokenize tools
special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
tokenizer = text_encoder(args.unit, special_tokens, args.corpus_path)
args.vocab_size = tokenizer.get_vocab_size()
id_to_token = tokenizer.id_to_token

# init model 
las =  LAS(args, Listener, Speller, id_to_token)

# build batch iterator
print("Build batch iterator...")
train_iter, num_train_batches = batch_gen(
                                    train_feats, 
                                    train_tokens, 
                                    train_featlen, 
                                    train_tokenlen, 
                                    args.batch_size, 
                                    args.feat_dim, 
                                    args.bucketing, 
                                    is_training=True)
dev_iter, num_dev_batches = batch_gen(
                                    dev_feats, 
                                    dev_tokens, 
                                    dev_featlen, 
                                    dev_tokenlen, 
                                    args.batch_size, 
                                    args.feat_dim, 
                                    True, 
                                    is_training=False)
train_xs, train_ys = train_iter.get_next()
dev_xs, dev_ys = dev_iter.get_next()

# build train, inference graph 
print("Build train, inference graph (please wait)...")
loss, train_op, global_step, train_logits, alphas, train_summary = las.train(train_xs, train_ys)
dev_logits, y_hat = las.inference(dev_xs)

# saver
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
if not os.path.exists(args.corpus_path):
    os.makedirs(args.corpus_path)
saver = tf.train.Saver(max_to_keep=100)

if args.restore > 0:
    ckpt = args.save_path + "/las_E{}".format(args.restore)
else:
    ckpt = tf.train.latest_checkpoint(args.save_path)

if ckpt is None:
    sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, ckpt)

# init iterator and graph
sess.run(train_iter.initializer)
sess.run(dev_iter.initializer)

# summary
summary_writer = tf.summary.FileWriter(args.summary_path, sess.graph)

# info
print("INFO: Training command:"," ".join(sys.argv))
print("INFO: Total weights:", 
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

# training
training_steps = num_train_batches * args.epoch
print("INFO: Total num train batches:", num_train_batches)
print("Training...")
loss_ = []
for step in range(training_steps):
    batch_loss, gs, _, summary_, logits, train_gt = sess.run(
                        [loss, global_step, train_op, train_summary, train_logits, train_ys])

    print("HYP: {}\nREF: {}".format(
        convert_idx_to_string(np.argmax(logits, -1)[0], id_to_token, args.unit), 
        convert_idx_to_string(train_gt[0][0], id_to_token, args.unit)))

    print("INFO: num_step: {}, loss: {}".format(gs, batch_loss))
    summary_writer.add_summary(summary_, gs)
    loss_.append(batch_loss)
    if gs and gs % num_train_batches == 0:
        ave_loss = np.mean(loss_)
        e_ =  gs // num_train_batches
        print("INFO: num epoch: {}, num_step: {}, ave loss: {}, wer: {}".format(
                                                e_, gs, ave_loss, 0))
        saver.save(sess, args.save_path+"/las_E{}".format(e_))      
        loss_ = []  

summary_writer.close()
