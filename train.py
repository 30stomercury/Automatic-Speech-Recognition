from las.utils import *
from las.arguments import *
from las.las import *
from data_loader import *
import os
import sys
import numpy as np
import tensorflow as tf


# arguments
args = parse_args()

# init session 
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# define data generator
train_libri_path = args.train_data_path
dev_libri_path = args.dev_data_path
train_texts, train_audio_path = data_preparation(train_libri_path)
dev_texts, dev_audio_path = data_preparation(dev_libri_path)

# load from previous output
try:
    print("Load features...")
    train_feats = np.load(args.feat_path+"/train_feats.npy", allow_pickle=True)
    train_featlen = np.load(args.feat_path+"/train_featlen.npy", allow_pickle=True)
    train_chars = np.load(args.feat_path+"/train_chars.npy", allow_pickle=True)
    train_charlen = np.load(args.feat_path+"/train_charlen.npy", allow_pickle=True)
    dev_feats = np.load(args.feat_path+"/dev_feats.npy", allow_pickle=True)
    dev_featlen = np.load(args.feat_path+"/dev_featlen.npy", allow_pickle=True)
    dev_chars = np.load(args.feat_path+"/dev_chars.npy", allow_pickle=True)
    dev_charlen = np.load(args.feat_path+"/dev_charlen.npy", allow_pickle=True)
    special_chars = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
    char2id, id2char = lookup_dicts(special_chars)

# process features
except:
    raise Exception("Run preprocess.py first")

# Clip text length to predefined decoding steps
# train
index = train_charlen < args.maxlen
train_feats = train_feats[index]
train_featlen = train_featlen[index]
train_chars = train_chars[index]
train_charlen = train_charlen[index]

# init model 
args.vocab_size = len(char2id)
if args.ctc:
    args.vocab_size += 1
las =  LAS(args, Listener, Speller, char2id, id2char)

# build batch iterator
print("Build batch iterator...")
train_iter, num_train_batches = batch_gen(
            train_feats, train_chars, train_featlen, train_charlen, args.batch_size, args.feat_dim, args.bucketing, shuffle_batches=True)
dev_iter, num_dev_batches = batch_gen(
            dev_feats, dev_chars, dev_featlen, dev_charlen, args.batch_size, args.feat_dim, True, shuffle_batches=False)
train_xs, train_ys = train_iter.get_next()
dev_xs, dev_ys = dev_iter.get_next()

# build train, inference graph 
print("Build train, inference graph (please wait)...")
loss, train_op, global_step, train_logits, alphas, train_summary = las.train(train_xs, train_ys)
dev_logits, y_hat = las.inference(dev_xs)
sample = convert_idx_to_token_tensor(y_hat[0], id2char)

# saver
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
saver = tf.train.Saver(max_to_keep=100)
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
print("INFO: Total weights:",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
#print(tf.trainable_variables())

# training
training_steps = num_train_batches * args.epoch
print("INFO: Total num train batches:", num_train_batches)
print("Training...")
loss_ = []
for step in range(training_steps):
    batch_loss, gs, _, summary_, logits, train_gt = sess.run([loss, global_step, train_op, train_summary, train_logits, train_ys])
    print("Sample: {}\nGround: {}".format(convert_idx_to_string(np.argmax(logits, -1)[0], id2char), convert_idx_to_string(train_gt[0][0], id2char)))
    print("INFO: num_step: {}, loss: {}".format(gs, batch_loss))
    summary_writer.add_summary(summary_, gs)
    loss_.append(batch_loss)
    if gs and gs % num_train_batches == 0:
        ave_loss = np.mean(loss_)
        e_ =  gs // num_train_batches
        print("INFO: num epoch: {}, num_step: {}, ave loss: {}, wer: {}".format(
                                                e_, gs, ave_loss, 0))
        saver.save(sess, args.save_path+"/las_E{}".format(e_), global_step=gs)      
        loss_ = []  
        # eval
        print("Inference...")
        texts = get_texts(y_hat, sess, num_dev_batches, id2char) 
        with open(args.result_path+"/texts_E{}.txt".format(e_), 'w') as fout:
            fout.write("\n".join(texts))
    if gs % 50 == 0:
        sample_utt, gt = sess.run([sample, dev_ys])
        sample_utt = sample_utt.decode()
        gt_utt = convert_idx_to_string(gt[0][0], id2char)
        print("INFO: Sample utt | sample: {} | groundtruth: {}.".format(sample_utt, gt_utt))
summary_writer.close()
