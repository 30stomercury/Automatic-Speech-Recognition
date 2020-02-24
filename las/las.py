import numpy as np
import tensorflow as tf

from las.utils import *

class Listener:

    def __init__(self, args):
        self.args = args

    def __call__(self, inputs, audiolen, is_training=True):
        with tf.variable_scope("Listener", reuse=tf.AUTO_REUSE):
            enc_out, enc_state, enc_len = pblstm(
                inputs, audiolen, self.args.num_enc_layers, self.args.enc_units, self.args.dropout_rate, is_training)

            return enc_out, enc_state, enc_len

class Speller:

    def __init__(self, args):
        self.args = args
        self.hidden_size = self.args.enc_units*2 # => output dimension of plstm h
        self._build_decoder_cell()
        self._build_char_embeddings()

    def __call__(self, enc_out, enc_len, dec_steps, teacher=None, is_training=True):
        with tf.variable_scope("Speller", reuse=tf.AUTO_REUSE):

            if self.args.ctc:
                ctc_logits = tf.layers.dense(
                                enc_out, self.args.vocab_size+1)
            else:
                ctc_logits = None

            init_char = self._look_up(tf.ones(tf.shape(enc_out)[0], dtype=tf.int32))
            init_state = self.dec_cell.zero_state(tf.shape(enc_out)[0], tf.float32)
            init_t = tf.constant(0, dtype=tf.int32)
            init_output = tf.zeros([tf.shape(enc_out)[0], 1, self.args.vocab_size])
            init_alphas = tf.zeros([tf.shape(enc_out)[0], 1, tf.shape(enc_out)[1]])

            # define loop
            def iteration(t, rnn_state, prev_char, output, alphas):
                cur_char, rnn_state, alphas_ = \
                        self.decode(enc_out, enc_len, rnn_state, prev_char, is_training)
                if is_training:
                    condition = self.args.teacher_forcing_rate > tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
                    prev_char = tf.cond(condition,
                                        lambda: self._look_up(teacher[:, t]),           # => teacher forcing
                                        lambda: self._sample_char(cur_char))            # => or you can use argmax
                                               
                    prev_char = tf.layers.dropout(
                                prev_char, self.args.dropout_rate, training=is_training)
                    prev_char.set_shape([None, self.args.embedding_size])
                else:
                    prev_char = self._look_up(tf.argmax(cur_char, -1))                  # => inference mode, greedy decoder.

                cur_char = tf.expand_dims(cur_char, 1)
                output = tf.concat([output, cur_char], 1)
                alphas = tf.concat([alphas, tf.expand_dims(alphas_, 1)], 1)

                return t + 1, rnn_state, prev_char, output, alphas

            # stop criteria
            def is_stop(t, *args):
                return t < dec_steps

            # define shape of tensors in iteration
            if self.args.num_dec_layers > 1:
                shape_state = ()
                for l in range(self.args.num_dec_layers): shape_state += (tf.TensorShape([None, None]),) 
            else:
                shape_state = tf.TensorShape([None, None])

            shape_invariant = [init_t.get_shape(), 
                               shape_state,
                               init_char.get_shape(), 
                               tf.TensorShape([None, None, self.args.vocab_size]),
                               tf.TensorShape([None, None, None])]

            t, dec_state, prev_char, output, alphas = \
                        tf.while_loop(is_stop, iteration, 
                            [init_t, init_state, init_char, init_output, init_alphas], shape_invariant)
            
            logits = output[:, 1:, :]
            alphas = alphas[:, 1:, :]

            return logits, ctc_logits, alphas

    def decode(self, enc_out, enc_len, dec_state, prev_char, is_training):
        """One decode step."""
        with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
            s_i = self._get_hidden_state(dec_state)
            context, alphas = attention(h=enc_out, 
                                        state=s_i, 
                                        h_dim=self.hidden_size, 
                                        s_dim=self.args.dec_units*self.args.num_dec_layers,
                                        attention_size=self.args.attention_size,
                                        seq_len=enc_len)
            dec_in = tf.concat([prev_char, context], -1) # dim = h dim + embedding dim
            dec_out, dec_state = self.dec_cell(
                            dec_in, dec_state)
            cur_char = tf.layers.dense(
                            dec_out, 
                            self.args.vocab_size, use_bias=True)

            return cur_char, dec_state, alphas

    def _look_up(self, char):
        """lookup from pre-definded embedding"""
        return tf.nn.embedding_lookup(                                                                                                                                         
                    self.embedding_matrix, char)

    def _sample_char(self, char):
        """Sample charactor from char distribution."""
        dist = tf.distributions.Categorical(logits=char)
        sampled_char = dist.sample(int(1))[0]

        return self._look_up(sampled_char)

    def _get_hidden_state(self, dec_state):
        if self.args.num_dec_layers > 0:
            return tf.concat(dec_state, -1)
        else:
            return dec_state[1]
            
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

    def __init__(self, args, listener, speller, id_to_token):
        '''Consturct Listen attend ande spell objects.
        Args:
            args: Include model/train/inference parameters that are packed in arguments.py
            listener: The encoder built in pyramid rnn.
            speller: The decoder with an attention layer.
        '''
        self.args = args
        self.listener = Listener(args)
        self.speller = Speller(args)
        self.id_to_token = id_to_token

    def train(self, xs, ys):
        ''' Buid decoder encoder network, compute loss.
        Args:
            xs: a tuple of 
                - audio:    (B, T1, D), T1: padded feature timesteps, D: feature dimension.
                - audiolen: (B,), original feature length.
            ys: 
                - y:        (B, T2), T2: output time steps.
                - charlen:  (B,), original character length.
        Returns: 
            loss
            train_op
            global_step
            summaries
        '''
        audio, audiolen = xs
        y, charlen = ys
        # encoder decoder network
        dec_steps = tf.shape(y)[1] # => time steps in this batch
        h, enc_state, enc_len = self.listener(audio, audiolen)
        logits, ctc_logits, alphas = self.speller(h, enc_len, dec_steps, y)

        # Scale to [0, 255]
        attention_images = alphas*255

        # compute loss
        att_loss = self._get_loss(logits, y, charlen)
        if self.args.ctc:
            ctc_loss = self._get_ctc_loss(ctc_logits, y, enc_len)
            loss = self.args.ctc_weight*ctc_loss + att_loss
        else:
            loss = att_loss
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(self.args.lr)
        # gradient clipping
        if self.args.grad_clip > 0:
            grad, variables = zip(*optimizer.compute_gradients(loss))
            grad, _ = tf.clip_by_global_norm(grad, self.args.grad_clip)
            train_op = optimizer.apply_gradients(zip(grad, variables), global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)
        # sample one utterance
        sample = convert_idx_to_token_tensor(
                    tf.argmax(logits, -1)[0], self.id_to_token, self.args.unit)
        ref = convert_idx_to_token_tensor(
                    y[0], self.id_to_token, self.args.unit)
        # summary
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)
        tf.summary.text("sample_prediction", sample)
        tf.summary.text("ground_truth", ref)
        tf.summary.image('attention_images', tf.expand_dims(attention_images, -1))
        tf.summary.image('features', tf.expand_dims(audio, -1))

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, logits, alphas, summaries

    def inference(self, xs):
        audio, audiolen = xs
        # estimate decoding steps
        dec_steps = tf.multiply(self.args.convert_rate, 
                                    tf.to_float(tf.reduce_max(audiolen)))
        dec_steps = tf.to_int32(dec_steps)
        # encoder decoder network
        h, enc_state, enc_len = self.listener(audio, audiolen, is_training=False)
        logits, ctc_logits, alphas  = self.speller(h, enc_len, dec_steps, is_training=False)
        y_hat = tf.argmax(logits, -1)

        return logits, y_hat
    
    def _get_loss(self, logits, y, charlen):
        if self.args.label_smoothing:
            y_ = label_smoothing(tf.one_hot(y, depth=self.args.vocab_size))
        else:
            y_ = tf.one_hot(y, self.args.vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
        mask_padding = 1 - tf.cast(tf.equal(y, 0), tf.float32)
        loss = tf.reduce_sum(
            cross_entropy * mask_padding) / (tf.reduce_sum(mask_padding) + 1e-9)
        return loss

    def _get_ctc_loss(self, ctc_logits, y, enc_len):
        labels = tf.cast(y, tf.int64)
        enc_len = tf.cast(enc_len, tf.int32)
        idx = tf.where(tf.not_equal(labels, 0))[:-1]
        sparse = tf.SparseTensor(idx, tf.gather_nd(labels, idx), tf.shape(labels, out_type=tf.int64))
        sparse = tf.cast(sparse, tf.int32)
        return tf.reduce_mean(
                tf.nn.ctc_loss(
                        sparse,
                        inputs=ctc_logits,
                        sequence_length=enc_len,
                        preprocess_collapse_repeated=False,
                        ctc_merge_repeated=True,
                        ignore_longer_outputs_than_inputs=False,
                        time_major=False))
