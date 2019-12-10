import numpy as np
import tensorflow as tf

from las.utils import *

class Listener:

    def __init__(self, args):
        self.args = args

    def __call__(self, inputs, audio_len):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            enc_out, enc_state, enc_len = pblstm(
                inputs, audio_len, self.args.num_enc_layers, self.args.enc_units, self.args.keep_proba, self.args.is_training)
        return enc_out, enc_state, enc_len

class Speller:

    def __init__(self, args, char2id):
        self.args = args
        self.char2id = char2id
        self.hidden_size \
                = self.args.enc_units*2*(self.args.num_enc_layers*2)**2 # - > output dimension of plstm h
        self._build_decoder_cell()
        self._build_char_embeddings()

    def __call__(self, enc_out, enc_len, dec_steps, teacher):

        prev_char = tf.nn.embedding_lookup(
                        self.embedding_matrix, tf.ones(tf.shape(enc_out)[0], dtype=tf.int32))
        dec_state = self.dec_cell.zero_state(tf.shape(enc_out)[0], tf.float32)
        init_t = tf.constant(0, dtype=tf.int32)
        maxlen = tf.shape(teacher)[1]
        output = tf.zeros([tf.shape(enc_out)[0], 1, self.args.vocab_size])

        def iteration(t, dec_state, prev_char, output):
            cur_char, dec_state, alphas = self.decode(enc_out, enc_len, dec_state, prev_char)
            if self.args.is_training:
                condition = self.args.teacher_forcing_rate < tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
                prev_char = tf.cond(condition,
                                    lambda: self.teacher_forcing(teacher[:, t]), # => teacher forcing
                                    lambda: self.sample(cur_char))               # => sample from categorical distribution
                prev_char = tf.nn.dropout(prev_char, self.args.keep_proba)
                prev_char.set_shape([None, self.args.embedding_size])
            else:
                prev_char = tf.nn.embedding_lookup(
                        self.embedding_matrix, tf.argmax(cur_char, -1))

            cur_char = tf.expand_dims(cur_char, 1)
            output = tf.concat([output, cur_char], 1)

            return t + 1, dec_state, prev_char, output

        def is_stop(t, *args):
            return t < maxlen
        # define shape of tensors in iteration
        shape_state = ()
        for l in range(self.args.num_dec_layers): shape_state += (tf.TensorShape([None, None]),) 
        shape_invariant = [init_t.get_shape(), 
                           shape_state,
                           prev_char.get_shape(), 
                           tf.TensorShape([None, None, self.args.vocab_size])]

        t, dec_state, prev_char, output = \
                    tf.while_loop(is_stop, iteration, [init_t, dec_state, prev_char, output], shape_invariant)
        
        logits = output[:, 1:, :]

        return logits

    def decode(self, enc_out, enc_len, dec_state, prev_char):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            context, alphas = attention(h=enc_out, 
                                        char=prev_char, 
                                        hidden_size=self.hidden_size, 
                                        embedding_size=self.args.embedding_size, 
                                        seq_len=enc_len)
            dec_in = tf.concat([prev_char, context], -1) # dim = h dim + embedding dim
            dec_out, dec_state = self.dec_cell(
                            dec_in, dec_state)
            cur_char = tf.layers.dense(
                            dec_out, 
                            self.args.vocab_size, use_bias=True)
            cur_char = tf.nn.dropout(cur_char, self.args.keep_proba)
            return cur_char, dec_state, alphas
   
    def teacher_forcing(self, cur_char):
        return tf.nn.embedding_lookup(
                    self.embedding_matrix, cur_char)

    def sample(self, cur_char):
        """Sample charactor from char distribution."""
        dist = tf.distributions.Categorical(logits=cur_char)
        sample = dist.sample(int(1))[0]
        return tf.nn.embedding_lookup(
                    self.embedding_matrix, sample)

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
        self.listener = Listener(args)
        self.speller = Speller(args, char2id)
        self.char2id = char2id
        self.id2char = id2char

    def train(self, xs, ys):
        ''' Buid decoder encoder network, compute loss.
        Args:
            xs: a tuple of 
                - audio:    (B, T1, D), T1: padded feature timesteps.
                - audiolen: (B,), original feature length.
            ys: 
                - y:        (B, T2), T2: output time steps.
                - charlen:  (B,), original character length.
        Returns: 
            loss
            train_op
            global_step
        '''
        audio, audiolen = xs
        y, charlen = ys
        # training phase
        self.listener.args.is_training = True
        self.speller.args.is_training = True
        # encoder decoder network
        h, enc_state, enc_len = self.listener(audio, audiolen)
        logits  = self.speller(h, enc_len, self.args.dec_steps, y)
        # compute loss
        loss = self.get_loss(logits, y, charlen)
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(self.args.lr)
        # gradient clipping
        if self.args.grad_clip > 0:
            grad, variables = zip(*optimizer.compute_gradients(loss))
            grad, _ = tf.clip_by_global_norm(grad, self.args.grad_clip)
            train_op = optimizer.apply_gradients(zip(grad, variables), global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)
        return loss, train_op, global_step, logits

    def get_loss(self, logits, y, charlen):
        y_ = tf.one_hot(y, self.args.vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        mask_padding = 1 - tf.cast(tf.equal(y, 0), tf.float32)
        loss = tf.reduce_sum(
            cross_entropy * mask_padding) / (tf.reduce_sum(mask_padding) + 1e-9)
        return loss

    def inference(self, xs, ys):
        audio, audiolen = xs
        y, charlen = ys
        # inference phase
        self.listener.args.is_training = False
        self.speller.args.is_training = False
        # encoder decoder network
        h, enc_state, enc_len = self.listener(audio, audiolen)
        logits  = self.speller(h, enc_len, self.args.dec_steps, y)
        y_hat = tf.argmax(logits, -1)

        return logits, y_hat

    def debug(self, xs, ys):
        audio, audio_len = xs
        y, char_len = ys
        # encoder decoder network
        h, enc_state, enc_len = self.listener(audio, audiolen)
        logits, alphas = self.speller(
                            h, enc_len, y.shape[-1], y)
        return logits
