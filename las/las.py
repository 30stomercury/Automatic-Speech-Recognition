import numpy as np
import tensorflow as tf

from las import utils

class Listener:

    def __init__(self, args):
        self.args = args

    def __call__(self, inputs, audio_len):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            enc_out, enc_states, enc_len = pblstm(inputs, audio_len, self.args.num_enc_layers, self.args.enc_units, self.args.keep_proba, self.args.is_training)
        return enc_out, enc_state, enc_len

class Speller:

    def __init__(self, args):
        self.args = args
        self._build_decoder_cell()
        self._build_char_embeddings()

    def __call__(self, enc_out, enc_len, dec_step, teacher):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            prev_char = tf.nn.embedding_lookup(
                            self.embedding_matrix, tf.zeros(self.args.batch_size))
            dec_state = dec_cell.zero_state(self.args.batch_size, tf.float32)
            output = []
            alphas = []
            for t in range(dec_steps):
                context, alphas = self.attention(h=enc_out, 
                                                 char=prev_char, 
                                                 hidden_size=self.args.embedding_size, 
                                                 seqlength=enc_len)
                dec_in = tf.concat([prev_char, context], -1) # dim = enc dim + embedding dim
                dec_out, dec_state = self.dec_cell(
                                dec_in, dec_state)
                cur_char = tf.layers.dense(
                                self.dec_out, 
                                self.args.vocab_size, use_bias=True)
                cur_char = tf.nn.dropout(cur_char, self.args.keep_proba)
                if self.args.teacher_forcing:
                    prev_char = tf.nn.embedding_lookup(
                            embedding_matrix, teacher[:, t])
                else:
                    prev_char = tf.nn.embedding_lookup(
                            embedding_matrix, tf.argmax(cur_char, -1))
                prev_char = tf.nn.dropout(prev_char, self.args.keep_proba)

                output.append(cur_char)
                alphas.append(attn)
            logits = tf.stack(output, axis=1)
        return output, alphas

    def _build_decoder_cell(self):
        def rnn_cell():
            # lstm cell
            return tf.contrib.rnn.BasicRNNCell(self.args.dec_units)
        if self.args.num_dec_layers > 1:
            self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
                        [rnn_cell() for _ in range(self.args.num_dec_layers)], state_is_tuple=True)
        else:
            self.dec_cell = rnn_cell()

    def _build_char_embeddings(self):
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            embedding_matrix = tf.get_variable(
                name='embedding_matrix',
                shape=[self.args.vocab_size, self.args.embedding_size],
                initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.embedding_matrix = embedding_matrix

class LAS:

    def __init__(self, args, listener, speller, char2id, id2char):
        '''Consturct Listen attend ande spell objects.
        Args:
            args: Include model/train/inference parameters that are packed in arguments.py
        '''
        if not args.vocab_size:
            args.vocab = len(char2id)
        self.args = args
        self.listener = listener(args)
        self.speller = speller(args)
        self.char2id = char2id
        self.id2char = id2char

    def train(self, xs, ys):
        ''' Buid decoder encoder network, compute loss.
        Args:
            xs: a tuple of 
                - audio:     (B, T1, D), T1: padded feature timesteps.
                - audio_len: (B,), original feature length.
            ys: 
                - y:         (B, T2), T2: output time steps.
                - char_len:  (B,), original character length.
        Returns: 
            loss
            train_op
            global_step
        '''
        audio, audio_len = xs
        y, char_len = ys
        # encoder decoder network
        h, enc_state, enc_len = self.listener(audio, audio_len)
        logits, alphas = self.speller(
                            h, enc_len, dec_step, teacher)
        
        # compute loss
        loss = self.compute_loss(logits, y)
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(self.args.lr)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return loss, train_op, global_step

    def compute_loss(self, logits, labels):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
        mask_padding = mask(y, char_len)
        loss = tf.reduce_sum(cross_entropy * mask_padding) / (tf.reduce_sum(mask_padding) + 1e-9)
        return loss

