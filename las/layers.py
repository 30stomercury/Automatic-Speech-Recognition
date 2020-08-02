import tensorflow as tf
import numpy as np


def lstm(inputs, num_layers, cell_units, dropout_rate, is_training):
    def rnn_cell():
        # lstm cell
        return tf.contrib.rnn.BasicRNNCell(cell_units)
    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(
                [rnn_cell() for _ in range(num_layers)], state_is_tuple=True)
    else:
        cell = rnn_cell()
    # when training, add dropout to regularize.
    if is_training == True:
        keep_proba = 1 - dropout_rate
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                             input_keep_prob=keep_proba)
    else:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                             input_keep_prob=1)
    outputs, states = tf.nn.dynamic_rnn(cell, 
                                   inputs=inputs, 
                                   time_major=False,
                                   dtype=tf.float32) 
    return outputs, states

def blstm(inputs, cell_units, dropout_rate, is_training):
    def rnn_cell():
        # lstm cell
        return tf.contrib.rnn.BasicRNNCell(cell_units)
    # Forward direction cell
    fw_cell = rnn_cell()
    # Backward direction cell
    bw_cell = rnn_cell()
    # when training, add dropout to regularize.
    if is_training == True:
        keep_proba = 1 - dropout_rate
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,
                                                input_keep_prob=keep_proba)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,
                                                input_keep_prob=keep_proba)
    else:
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell,
                                                input_keep_prob=1)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell,
                                                input_keep_prob=1)

    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                      cell_bw=bw_cell,
                                                      inputs= inputs,
                                                      dtype=tf.float32,
                                                      time_major=False)
    return outputs, states

def PBLSTMLayer(inputs, audiolen, num_layers, cell_units, dropout_rate, is_training):
    """Pyramidal BLSTM
    blstm ->
    projection/tanh ->
    pblstm ->
    projection/tanh ->
    """
    
    batch_size = tf.shape(inputs)[0]
    enc_dim = cell_units*2
    # a BLSTM layer
    with tf.variable_scope('blstm'):
        rnn_out, _ = blstm(inputs, cell_units, dropout_rate, is_training)
        rnn_out = tf.concat(rnn_out, -1)        
        rnn_out = tf.layers.dense(
                            rnn_out,
                            enc_dim, use_bias=True, 
                            activation=tf.nn.tanh)

    # Pyramid BLSTM layers
    for l in range(num_layers):
        with tf.variable_scope('pyramid_blstm_{}'.format(l)):
            rnn_out, states = blstm(
                    rnn_out, cell_units, dropout_rate, is_training)
            rnn_out = tf.concat(rnn_out, -1)        
            # Eq (5) in Listen, Attend and Spell
            T = tf.shape(rnn_out)[1]
            padded_out = tf.pad(rnn_out, [[0, 0], [0, T % 2], [0, 0]])
            even_new = padded_out[:, ::2, :]
            odd_new = padded_out[:, 1::2, :]
            rnn_out = tf.concat([even_new, odd_new], -1)        
            rnn_out = tf.reshape(rnn_out, [batch_size, -1, cell_units*4])
            rnn_out = tf.layers.dense( 
                                rnn_out,
                                enc_dim, 
                                use_bias=True,
                                activation=tf.nn.tanh)
            audiolen = (audiolen + audiolen % 2) / 2
    return rnn_out, states, audiolen

def conv2d(inputs, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, stddev=1, name="conv2d", is_training=True):
    with tf.variable_scope(name) as scope:
        init = tf.random.normal(
                [k_h, k_w, inputs.get_shape().as_list()[-1], output_dim], stddev=stddev)*0.001
        w = tf.get_variable('w', initializer=init)
        b = tf.get_variable(
            'b', [output_dim], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(inputs, w, strides=[1, d_h, d_w, 1], padding='SAME')
        conv += b
        conv = bn(conv, is_training)
        conv = tf.nn.relu(conv)

        return conv

def CNNLayer(inputs, audiolen,  feat_dim, cell_units, num_channel, dropout_rate, is_training):
    """CNN networks 
    cnn/batch-norm/relu ->
    cnn/batch-norm/relu ->
    blstm ->
    projection/batch-norm/relu ->
    blstm ->
    projection/batch-norm/relu ->
    """

    enc_units = cell_units

    # cnn layers
    conv_out = inputs 
    for i in range(2):
        # time & feat_dim reduction
        feat_dim = (feat_dim + feat_dim % 2) / 2
        audiolen = (audiolen + audiolen % 2) / 2

        # output_shape = [None, audiolen, feat_dim, num_channel]
        conv_out = conv2d(conv_out, 
                          num_channel, 
                          name="conv2d_{}".format(i), is_training=is_training)

    # reshape to [B, L/4, feat_dim/4 * 32]
    shape = tf.shape(conv_out)
    enc_out = tf.reshape(conv_out, [shape[0], -1, int(feat_dim*num_channel)])

    # blstm layers
    for i in range(self.args.num_enc_layers):
        with tf.variable_scope('blstm_{}'.format(i)):
            enc_out, enc_state = blstm(enc_out, cell_units, dropout_rate, is_training)
            enc_out = tf.concat(enc_out, -1)        
            enc_out = tf.layers.dense(
                             enc_out,
                             enc_units, 
                             use_bias=True)
            enc_out = tf.nn.relu(bn(enc_out, is_training))

    return enc_out, enc_state, audiolen

class BaseAttention:
    """Base attention setup"""

    def __init__(self, att_size, smoothing):
        self.att_size = att_size
        self.smoothing = smoothing

    def mask(self, original_len, padded_len):
        """Creating mask for sequences with different lengths in a batch.

        Args:
            original_len: (B,), original lengths of sequences in a batch.
            padded_len:   scalar,  length of padded sequence.

        Return:
            mask:         (B, T), int32 tensor, a mask of varied lengths.

        For example:
            original_len = [2, 3, 1]
            padded_len = 3

            mask = ([1, 1, 0],
                    [1, 1, 1],
                    [1, 0, 0])
        """

        y = tf.range(1, padded_len+1, 1)
        original_len = tf.expand_dims(original_len, 1)
        original_len = tf.cast(original_len, tf.int32)
        y = tf.expand_dims(y, 0)
        mask = y <= original_len

        return tf.cast(mask, tf.float32)

    def attend(self, inputs, energy, seqlen):
        """Operate attention."""

        T = tf.shape(inputs)[1]

        # mask attention weights
        mask_att = self.mask(seqlen, T)   # (B, T)
        paddings = tf.ones_like(mask_att)*(-1e8)
        energy = tf.where(tf.equal(mask_att, 0), paddings, energy)
        alphas = tf.nn.softmax(energy)

        # Output reduced with context vector: (B, hidden_size)
        context = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        return context, alphas

class AdditiveAttention(BaseAttention):
    """Bahdanau attention

    See:

    "Neural Machine Translation by Jointly Learning to Align and Translate."
    https://arxiv.org/abs/1409.0473

    args:
        h_dim: encoder units.
        s_dim: decoder units.
        att_size: attention size.
    """

    def __init__(self, h_dim, s_dim, att_size, smoothing):
        super().__init__(att_size, smoothing)
        self.h_dim = h_dim
        self.s_dim = s_dim

    def __call__(self, hidden, state, align, seqlen):
        """
        args:
            hidden: (B, T, enc_units*2), encoder output.
            state: (B, dec_units), previous decoder hidden state.
            align: (B, T), needless, for completeness.
            seq_len: timesteps of sequences.
        """

        h.set_shape([None, None, self.h_dim])
        state = tf.reshape(state, [None, 1, self.s_dim])

        # Trainable parameters
        u = tf.Variable(tf.random_uniform([attention_size], minval=-1, maxval=1, dtype=tf.float32))
        v = tf.nn.tanh(
                tf.layers.dense(hidden, attention_size, use_bias=False) + \
                tf.layers.dense(state, attention_size, use_bias=False))
        energy = tf.tensordot(v, u, axes=1)


        context, alphas = self.attend(hidden, energy, seqlen)

        return context, alphas

class LocationAwareAttention(BaseAttention):
    """Location-aware attention

    See:

    "Attention-Based Models for Speech Recognition."
    https://arxiv.org/pdf/1506.07503.pdf

    args:
        h_dim: encoder units.
        s_dim: decoder units.
        att_size: attention size.
    """

    def __init__(self, h_dim, s_dim, att_size, kernel_size=10, num_channels=201, smoothing=False):

        super().__init__(att_size, smoothing)
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.kernel_size = kernel_size
        self.num_channels = num_channels

    def __call__(self, hidden, state, align, seqlen):
        """
        args:
            hidden: (B, T, enc_units), encoder output.
            state: (B, dec_units), previous decoder hidden state.
            align: (B, T), previous alignment.
            seq_len: timesteps of sequences.
        """

        h.set_shape([None, None, self.h_dim])
        state = tf.reshape(state, [-1, 1, self.s_dim])

        # conv location: eq (8)
        f = tf.layers.conv1d(tf.expand_dims(align, 2),
                             filters=self.num_chanel, kernel_size=self.kernel_size, strides=1, padding='SAME')

        u = tf.Variable(tf.random_uniform([attention_size], minval=-1, maxval=1, dtype=tf.float32))

        # eq (9)
        v = tf.nn.tanh(
                tf.layers.dense(h, attention_size, use_bias=False) + \
                tf.layers.dense(state, attention_size, use_bias=False) + \
                tf.layers.dense(f, attention_size, use_bias=False))
        energy = tf.tensordot(v, u, axes=1)


        context, alphas = self.attend(hidden, energy, seqlen)

        return context, alphas
