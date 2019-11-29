import tensorflow as tf

def mask(original_len, padded_len):
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

def attention(h, char, hidden_size, embedding_size, seq_len):
    """Bahdanau attention"""
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        attention_size = 16
        T = tf.shape(h)[1]
        char_tile = tf.tile(tf.expand_dims(char, 1), [1, T, 1])
        # Trainable parameters
        W_h = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        W_c = tf.Variable(tf.random_normal([embedding_size, attention_size], stddev=0.1))
        b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.nn.tanh(
                tf.tensordot(h, W_h, axes=1) + tf.tensordot(char_tile, W_c, axes=1) + b)
        vu = tf.tensordot(v, u, axes=1)
        # mask attention weights
        mask_att = mask(seq_len, T)   # (B, T)
        paddings = tf.ones_like(mask_att)*(-1e8)
        vu = tf.where(tf.equal(mask_att, 0), paddings, vu)      
        alphas = tf.nn.softmax(vu)                             
        # Output reduced with context vector: (B, T)
        out = tf.reduce_sum(h * tf.expand_dims(alphas, -1), 1)
    return out, alphas

def lstm(inputs, num_layers, cell_units, keep_proba, is_training):
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

def blstm(inputs, cell_units, keep_proba, is_training):
    def rnn_cell():
        # lstm cell
        return tf.contrib.rnn.BasicRNNCell(cell_units)
    # Forward direction cell
    fw_cell = rnn_cell()
    # Backward direction cell
    bw_cell = rnn_cell()
    # when training, add dropout to regularize.
    if is_training == True:
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

def pblstm(inputs, audio_len, num_layers, cell_units, keep_proba, is_training):
    batch_size = tf.shape(inputs)[0]
    # a BLSTM layer
    with tf.variable_scope('blstm'):
        rnn_out, _ = blstm(inputs, cell_units, keep_proba, is_training)
        rnn_out = tf.concat(rnn_out, -1)        
        cell_units = cell_units*2
    # Pyramid BLSTM layers
    for l in range(num_layers):
        with tf.variable_scope('pyramid_blstm_{}'.format(l)):
            rnn_out, states = blstm(
                    rnn_out, cell_units, keep_proba, is_training)
            rnn_out = tf.concat(rnn_out, -1)        
            # Eq (5) in Listen, Attend and Spell
            T = tf.shape(rnn_out)[1]
            padded_out = tf.pad(rnn_out, [[0, 0], [0, T % 2], [0, 0]])
            even_new = padded_out[:, ::2, :]
            odd_new = padded_out[:, 1::2, :]
            rnn_out = tf.concat([even_new, odd_new], -1)        
            cell_units = cell_units*4
            rnn_out = tf.reshape(rnn_out, [batch_size, -1, cell_units])
            audio_len = (audio_len + audio_len % 2) / 2
    return rnn_out, states, audio_len
