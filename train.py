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
sess = tf.Session()

# define data generator
train_libri_path = './data/LibriSpeech_train/train-clean-100'
dev_libri_path = './data/LibriSpeech_dev/dev-clean'
train_texts, train_audio_path = data_preparation(train_libri_path)
dev_texts, dev_audio_path = data_preparation(dev_libri_path)

# process features
print("Process train/dev features...")
special_chars = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
train_chars, train_charlen, char2id, id2char = process_texts(special_chars, train_texts)
dev_chars, dev_charlen, _, _ = process_texts(special_chars, dev_texts)
train_feats, train_featlen = process_audio(train_audio_path, 
                                           sess, 
                                           prepro_batch=64,
                                           sample_rate=args.sample_rate,
                                           frame_step=args.frame_step,
                                           frame_length=args.frame_length,
                                           feat_dim=args.feat_dim,
                                           feat_type=args.feat_type)
dev_feats, dev_featlen = process_audio(dev_audio_path, 
                                         sess, 
                                         prepro_batch=64,
                                         sample_rate=args.sample_rate,
                                         frame_step=args.frame_step,
                                         frame_length=args.frame_length,
                                         feat_dim=args.feat_dim,
                                         feat_type=args.feat_type)

# set decoding steps to max length and break if all cur_char in a batch are predicted to <PAD>
# see line 24 in las/las.py
args.dec_steps = np.max(np.max(train_charlen) + np.max(dev_charlen))

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
logits, y_hat, atten = las.inference(dev_xs, dev_ys)
sample = convert_idx_to_token_tensor(y_hat[0], id2char)

# saver
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
saver = tf.train.Saver(max_to_keep=100)
ckpt = tf.train.latest_checkpoint(args.save_path)
if ckpt is None:
    sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, ckpt)

# init iterator and graph
sess.run(tf.global_variables_initializer())
sess.run(train_iter.initializer)
sess.run(dev_iter.initializer)

print(sess.run(train_xs)[0].shape, sess.run(train_ys))
print(sess.run(logits).shape)
print(sess.run(1 - tf.cast(tf.equal(train_ys[0], 0), tf.float32)).shape)

# training
training_steps = num_train_batches * args.epoch
print("num train batch", num_train_batches)
loss_ = []
for step in range(training_steps):
    batch_loss, gs, _ = sess.run([loss, global_step, train_op])
    print(step, gs, training_steps)
    loss_.append(batch_loss)
    if gs and gs % num_train_batches == 0:
        ave_loss = np.mean(loss_)
        e_ =  gs // num_train_batches
        print("num epoch: {}, num_step: {}, ave loss: {}, wer: {}".format(
                                                e_, gs, ave_loss, 0))
        saver.save(sess, args.save_path+"/las_E{}".format(e_), global_step=_gs)        
        
        # eval
        texts = get_texts(y_hat, sess, num_dev_batches) 
        with open(args.results_path+"/texts_E{}".format(e_), 'w') as fout:
            fout.write("\n".join(texts))
