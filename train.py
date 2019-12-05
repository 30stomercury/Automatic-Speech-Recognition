from las.utils import *
from las.arguments import *
from las.las import *
from data_loader import *
import os
import numpy as np
import tensorflow as tf


# arguments
args = parse_args()

# init session 
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# define data generator
train_libri_path = './data/LibriSpeech_train/train-clean-100'
dev_libri_path = './data/LibriSpeech_dev/dev-clean'
train_texts, train_audio_path = data_preparation(train_libri_path)
dev_texts, dev_audio_path = data_preparation(dev_libri_path)

# load from previous output
try:
    print("Load features...")
    train_feats = np.load("data/features/train_feats.npy", allow_pickle=True)
    train_featlen = np.load("data/features/train_featlen.npy", allow_pickle=True)
    train_chars = np.load("data/features/train_chars.npy", allow_pickle=True)
    train_charlen = np.load("data/features/train_charlen.npy", allow_pickle=True)
    dev_feats = np.load("data/features/dev_feats.npy", allow_pickle=True)
    dev_featlen = np.load("data/features/dev_featlen.npy", allow_pickle=True)
    dev_chars = np.load("data/features/dev_chars.npy", allow_pickle=True)
    dev_charlen = np.load("data/features/dev_charlen.npy", allow_pickle=True)

    special_chars = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
    char2id, id2char = lookup_dicts(special_chars)

# process features
except:
    print("Process train/dev features...")
    special_chars = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
    train_chars, train_charlen, char2id, id2char = process_texts(special_chars, train_texts)
    dev_chars, dev_charlen, _, _ = process_texts(special_chars, dev_texts)
    train_feats, train_featlen = process_audio(train_audio_path, 
                                               sess, 
                                               prepro_batch=100,
                                               sample_rate=args.sample_rate,
                                               frame_step=args.frame_step,
                                               frame_length=args.frame_length,
                                               feat_dim=args.feat_dim,
                                               feat_type=args.feat_type)
    dev_feats, dev_featlen = process_audio(dev_audio_path, 
                                           sess, 
                                           prepro_batch=100,
                                           sample_rate=args.sample_rate,
                                           frame_step=args.frame_step,
                                           frame_length=args.frame_length,
                                           feat_dim=args.feat_dim,
                                           feat_type=args.feat_type)
    np.save("data/train_feats.npy", train_feats)    
    np.save("data/train_featlen.npy", train_featlen)
    np.save("data/dev_feats.npy", dev_feats)
    np.save("data/dev_featlen.npy", dev_featlen)
    np.save("data/train_chars.npy", train_chars)
    np.save("data/train_charlen.npy", train_charlen)
    np.save("data/dev_chars.npy", dev_chars)
    np.save("data/dev_charlen.npy", dev_charlen)

# Clip text length to predefined decoding steps
# train
index = train_charlen < args.dec_steps
train_feats = train_feats[index]
train_featlen = train_featlen[index]
train_chars = train_chars[index]
train_charlen = train_charlen[index]
# dev
index = dev_charlen < args.dec_steps
dev_feats = dev_feats[index]
dev_featlen = dev_featlen[index]
dev_chars = dev_chars[index]
dev_charlen = dev_charlen[index]

# init model 
las =  LAS(args, Listener, Speller, char2id, id2char)

# build batch iterator
print("Build batch iterator...")
train_iter, num_train_batches = batch_gen(
            train_feats, train_chars, train_featlen, train_charlen, args.batch_size, args.feat_dim, args.dec_steps, shuffle=True)
dev_iter, num_dev_batches = batch_gen(
            dev_feats, dev_chars, dev_featlen, dev_charlen, args.batch_size, args.feat_dim, args.dec_steps, shuffle=False)
train_xs, train_ys = train_iter.get_next()
dev_xs, dev_ys = dev_iter.get_next()

# build train, inference graph 
print("Build train, inference graph (please wait)...")
loss, train_op, global_step, logits = las.train(train_xs, train_ys)
logits, y_hat  = las.inference(dev_xs, dev_ys)
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

# training
training_steps = num_train_batches * args.epoch
print("Total num train batches:", num_train_batches)
loss_ = []
for step in range(training_steps):
    batch_loss, gs, _ = sess.run([loss, global_step, train_op])
    print("num_step: {}, loss: {}".format(gs, batch_loss))
    loss_.append(batch_loss)
    if gs and gs % num_train_batches == 0:
        ave_loss = np.mean(loss_)
        e_ =  gs // num_train_batches
        print("num epoch: {}, num_step: {}, ave loss: {}, wer: {}".format(
                                                e_, gs, ave_loss, 0))
        saver.save(sess, args.save_path+"/las_E{}".format(e_), global_step=gs)        
        
        # eval
        texts = get_texts(y_hat, sess, num_dev_batches, id2char) 
        with open(args.result_path+"/texts_E{}.txt".format(e_), 'w') as fout:
            fout.write("\n".join(texts))
