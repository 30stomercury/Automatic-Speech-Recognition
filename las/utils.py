import tensorflow as tf

def attention(h, char, hidden_size, seqlength):
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        ### context vector c w/ Bahdanau attention###
        attention_size = 16
        char_tile = tf.tile(tf.expand_dims(char, 1), [1, seqlength, 1])
        # Trainable parameters
        W_h = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        W_p = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.nn.tanh(tf.tensordot(h, W_h, axes=1) + tf.tensordot(char, W_p, axes=1) + b)
        vu = tf.tensordot(v, u, axes=1)
        # mask attention weights
        mask_att = tf.sign(tf.abs(tf.reduce_sum(h, axis=-1)))            # [B, seqlen]
        paddings = tf.ones_like(mask_att)*(-1e8)
        vu = tf.where(tf.equal(mask_att, 0), paddings, vu)               # (B, seqlen)
        alphas = tf.nn.softmax(vu)                                       # (B, seqlen)
        # Output reduced with context vector: [batch_size, sequence_len]
        out = tf.reduce_sum(h * tf.expand_dims(alphas, -1), 1)
    return out, alphas

def lstm(inputs, num_layers, cell_units, keep_proba, is_training):
    def rnn_cell():
        # lstm cell
        return tf.contrib.rnn.BasicRNNCell(cell_units)
    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell() for _ in range(num_layers)], state_is_tuple=True)
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

def pblstm(inputs, num_layers, cell_units, keep_proba, is_training):
    batch_size = tf.shape(inputs)[0]
    rnn_out = inputs
    for l in range(num_layers):
        with tf.variable_scope('pyramid_blstm_{}'.format(l)):
            rnn_out, states = blstm(
                    rnn_out, cell_units, keep_proba, is_training)
            rnn_out = tf.concat(rnn_out, -1)        
            # Eq (5) in Listen, Attend and Spell
            seqlength = tf.shape(rnn_out)[1]
            padded_out = tf.pad(rnn_out, [[0, 0], [0, seqlength % 2], [0, 0]])
            even_new = padded_out[:, ::2, :]
            odd_new = padded_out[:, 1::2, :]
            rnn_out = tf.concat([even_new, odd_new], -1)        
            cell_units = cell_units*4
            rnn_out = tf.reshape(rnn_out, [batch_size, -1, cell_units])
    return rnn_out, states 
