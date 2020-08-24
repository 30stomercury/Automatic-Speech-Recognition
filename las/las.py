import numpy as np
import tensorflow as tf
from las.utils import label_smoothing, convert_idx_to_token_tensor
from las.layers import AdditiveAttention, LocationAwareAttention, pBLSTMLayer, CNNLayer

class Listener:

    def __init__(self, args):
        self.args = args

    def __call__(self, inputs, audiolen, encoder='cnn', is_training=True):
        with tf.variable_scope("Listener", reuse=tf.AUTO_REUSE):
            if encoder == 'pblstm':
                inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, self.args.feat_dim*3])
                enc_out, enc_state, enc_len = pBLSTMLayer(inputs, 
                                                          audiolen, 
                                                          self.args.num_enc_layers, 
                                                          self.args.enc_units, 
                                                          self.args.dropout_rate, 
                                                          self.args.apply_bn,
                                                          is_training)
            elif encoder == 'cnn':
                enc_out, enc_state, enc_len = CNNLayer(inputs, 
                                                       audiolen,
                                                       self.args.num_enc_layers, 
                                                       self.args.feat_dim,
                                                       self.args.enc_units, 
                                                       self.args.num_enc_channels, 
                                                       self.args.dropout_rate, 
                                                       self.args.apply_bn,
                                                       is_training)

            else:
                raise NotImplementedError

            return enc_out, enc_state, enc_len


class Attention:

    def __init__(self, h_dim, s_dim, att_size, kernel_size, num_channels, mode='add'):
        self.mode = mode

        if self.mode == 'add':
            self.att_layer = AdditiveAttention(h_dim, s_dim, att_size)
        elif self.mode == 'loc':
            self.att_layer = LocationAwareAttention(
                    h_dim, s_dim, att_size, kernel_size, num_channels)
        else:
            raise NotImplementedError

    def __call__(self, hidden, state, align, seqlen):

        return self.att_layer(hidden, state, align, seqlen)


class Speller:

    def __init__(self, args):
        self.args = args
        self.hidden_dim = self.args.enc_units  # => output dimension of encoder h
        self.state_dim = self.args.dec_units*self.args.num_dec_layers
        self._build_decoder_cell()
        self._build_embeddings()
        self.att_layer = Attention(h_dim=self.hidden_dim,                      # TODO move kernel_size, num_channels to args.py
                                   s_dim=self.state_dim,
                                   att_size=self.args.attention_size,
                                   kernel_size=self.args.loc_kernel_size,       # Refer to section 4.2 
                                   num_channels=self.args.loc_num_channels,     # in https://arxiv.org/pdf/1506.07503.pdf
                                   mode=self.args.mode)

    def __call__(self, enc_out, enc_len, dec_steps, teacher=None, is_training=True):
        with tf.variable_scope("Speller", reuse=tf.AUTO_REUSE):

            if self.args.ctc:
                ctc_logits = tf.layers.dense(
                                enc_out, self.args.vocab_size+1)
            else:
                ctc_logits = None

            init_token = self._look_up(tf.ones(tf.shape(enc_out)[0], dtype=tf.int32))
            init_state = self.dec_cell.zero_state(tf.shape(enc_out)[0], tf.float32)
            init_t = tf.constant(0, dtype=tf.int32)
            init_output = tf.zeros([tf.shape(enc_out)[0], 1, self.args.vocab_size])
            init_alphas = tf.zeros([tf.shape(enc_out)[0], 1, tf.shape(enc_out)[1]])

            if is_training and self.args.scheduled_sampling:
                tf_rate = self._scheduled_sampling()
            else:
                tf_rate = 0.9

            # define loop
            def iteration(t, rnn_state, prev_token, output, alphas):
                cur_token, rnn_state, alphas_ = self.decode(enc_out, 
                                                           enc_len, 
                                                           rnn_state, 
                                                           prev_token,                    # previous token
                                                           alphas[:, -1, :],              # previous alignment
                                                           is_training)
                if is_training:
                    condition = tf_rate > tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)

                    prev_token = tf.cond(condition,
                                        lambda: self._look_up(teacher[:, t]),             # => teacher forcing
                                        lambda: self._sample_token(cur_token))            # => or you can use argmax
                                               
                    prev_token = tf.layers.dropout(
                                prev_token, self.args.dropout_rate, training=is_training)
                    prev_token.set_shape([None, self.args.embedding_size])
                else:
                    prev_token = self._look_up(tf.argmax(cur_token, -1))                  # => inference mode, greedy decoder.

                cur_token = tf.expand_dims(cur_token, 1)
                output = tf.concat([output, cur_token], 1)
                alphas = tf.concat([alphas, tf.expand_dims(alphas_, 1)], 1)

                return t + 1, rnn_state, prev_token, output, alphas

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
                               init_token.get_shape(), 
                               tf.TensorShape([None, None, self.args.vocab_size]),
                               tf.TensorShape([None, None, None])]

            t, dec_state, prev_token, output, alphas = \
                        tf.while_loop(is_stop, iteration, 
                            [init_t, init_state, init_token, init_output, init_alphas], shape_invariant)
            
            logits = output[:, 1:, :]
            alphas = alphas[:, 1:, :]

            return logits, ctc_logits, alphas

    def decode(self, enc_out, enc_len, dec_state, prev_token, prev_align, is_training):
        """One decode step."""
        with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
            s_i = self._get_hidden_state(dec_state)
            context, alphas = self.att_layer(hidden=enc_out, 
                                             state=s_i, 
                                             align=prev_align,
                                             seqlen=enc_len)
            dec_in = tf.concat([prev_token, context], -1) # dim = h dim + embedding dim
            dec_out, dec_state = self.dec_cell(
                            dec_in, dec_state)
            cur_token = tf.layers.dense(
                            dec_out, 
                            self.args.vocab_size, use_bias=True)

            return cur_token, dec_state, alphas

    def _look_up(self, token):
        """lookup from pre-definded embedding"""
        if self.args.add_vn:
            return tf.nn.embedding_lookup(self.embedding_matrix + \
                        tf.random.normal(tf.shape(self.embedding_matrix), stddev=0.075), token)

        return tf.nn.embedding_lookup(self.embedding_matrix, token)

    def _sample_token(self, token):
        """Sample token from token distribution."""
        dist = tf.distributions.Categorical(logits=token)
        sampled_token = dist.sample(int(1))[0]

        return self._look_up(sampled_token)

    def _scheduled_sampling(self):
        """Scheduled sampling with linear decay."""
        step = tf.train.get_or_create_global_step()
        step = tf.cast(step, tf.float32)
        progress = tf.minimum(
            (step-self.args.warmup_step) / float(self.args.max_step-self.args.warmup_step), 1.0)
        return tf.minimum(1.0, 1.0 - progress * (1.0 -self.args.min_rate))

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

    def _build_embeddings(self):
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            embedding_matrix = tf.get_variable(
                name='embedding_matrix',
                shape=[self.args.vocab_size, self.args.embedding_size],
                initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.embedding_matrix = embedding_matrix

class LAS:

    def __init__(self, args, Listener, Speller, id_to_token):
        '''Consturct Listen attend ande spell objects.

        Args:
            args:     Include model/train/inference parameters that are packed in arguments.py
            Listener: The encoder built in pyramid rnn.
            Attention: The Attention layer.
            Speller:  The decoder with an attention layer.
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
                - tokenlen: (B,), original token length.

        Returns: 
            loss
            train_op
            global_step
            summaries
        '''

        audio, audiolen = xs
        y, tokenlen = ys

        # encoder decoder network
        dec_steps = tf.reduce_max(tokenlen) # => time steps in this batch
        enc_type = self.args.enc_type.lower()
        h, enc_state, enc_len = self.listener(audio, audiolen, enc_type)
        logits, ctc_logits, alphas = self.speller(h, enc_len, dec_steps, y)

        # Scale to [0, 255]
        attention_images = alphas*255

        # compute loss
        att_loss = self._get_loss(logits, y)

        if self.args.ctc:
            ctc_loss = self._get_ctc_loss(ctc_logits, y, enc_len)
            loss = self.args.ctc_weight*ctc_loss + att_loss
        else:
            loss = att_loss

        global_step = tf.train.get_or_create_global_step()
        
        # update moving_mean & moving_variance 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        lr =  self._scheduled_learning_rate(
                start=10000, decay_step=30000, decay_rate=0.5, min_rate=0.01, global_step=global_step)
        optimizer = tf.train.AdamOptimizer(lr)

        # gradient clipping
        grad, variables = zip(*optimizer.compute_gradients(loss))
        grad, _ = tf.clip_by_global_norm(grad, self.args.grad_clip)

        with tf.control_dependencies(update_ops):
            if self.args.grad_clip > 0:
                    train_op = optimizer.apply_gradients(
                            zip(grad, variables), global_step=global_step)
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

        # sample rate
        sample_rate = self.speller._scheduled_sampling()

        return loss, train_op, global_step, logits, alphas, summaries, sample_rate

    def inference(self, xs):

        audio, audiolen = xs
        # estimate decoding steps
        dec_steps = tf.multiply(self.args.convert_rate, 
                                    tf.to_float(tf.reduce_max(audiolen)))
        dec_steps = tf.to_int32(dec_steps)
        # encoder decoder network
        h, enc_state, enc_len = self.listener(audio, audiolen, encoder='cnn', is_training=False)
        logits, ctc_logits, alphas  = self.speller(h, enc_len, dec_steps, is_training=False)
        y_hat = tf.argmax(logits, -1)

        return logits, y_hat
    
    def _get_loss(self, logits, y):
        y = tf.slice(y, [0, 0], tf.shape(logits)[:2]) # crop padding
        y_ = tf.one_hot(y, self.args.vocab_size)

        if self.args.label_smoothing:
            y_ = label_smoothing(y_)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                            logits=logits, labels=y_)
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

    def _scheduled_learning_rate(
            self, start=50000, decay_step=100000, decay_rate=0.5, min_rate=0.01, global_step=0):
        '''Using scheduled learning rate in: https://arxiv.org/pdf/1904.08779.pdf

        Args:
            start:      int, start learning rate decay.
            decay_step: int, decay constant.
            min_rate:   float, minimum learning rate.

        Return:
            scheduled_lr: scheduled learning rate.
        '''

        decayed_lr = tf.train.exponential_decay(self.args.lr,
                          tf.maximum(global_step-start, 0), 
                          decay_step, 
                          decay_rate)

        return tf.maximum(decayed_lr, min_rate*self.args.lr)
