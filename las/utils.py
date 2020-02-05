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

def label_smoothing(inputs, epsilon=0.1):
    """label smoothing
    Reference: https://github.com/Kyubyong/transformer/blob/master/tf1.2_legacy/modules.py
    """
    K = inputs.get_shape().as_list()[-1] 
    return ((1-epsilon) * inputs) + (epsilon / K)

def attention(h, state, h_dim, s_dim, attention_size, seq_len):
    """Bahdanau attention
    args:
        h: (B, T, enc_units*2), encoder output.
        state: (B, dec_units), previous decoder hidden state.
        h_dim: encoder units.
        s_dim: decoder units.
        seq_len: timesteps of sequences.
    """
    
    with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
        T = tf.shape(h)[1]
        # tiling to (B, T, dec_units)
        state_ = tf.expand_dims(state, 1)
        h.set_shape([None, None, h_dim])
        state_.set_shape([None, 1, s_dim])
        # Trainable parameters
        with tf.control_dependencies(None):
            u = tf.Variable(tf.random_uniform([attention_size], minval=-1, maxval=1, dtype=tf.float32))
        v = tf.nn.tanh(
                tf.layers.dense(h, attention_size, use_bias=False) + tf.layers.dense(state_, attention_size, use_bias=False))
        vu = tf.tensordot(v, u, axes=1)
        # mask attention weights
        mask_att = mask(seq_len, T)   # (B, T)
        paddings = tf.ones_like(mask_att)*(-1e8)
        vu = tf.where(tf.equal(mask_att, 0), paddings, vu)      
        alphas = tf.nn.softmax(vu)                             
        # Output reduced with context vector: (B, hidden_size)
        out = tf.reduce_sum(h * tf.expand_dims(alphas, -1), 1)
    return out, alphas

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

def pblstm(inputs, audio_len, num_layers, cell_units, dropout_rate, is_training):
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
            audio_len = (audio_len + audio_len % 2) / 2
    return rnn_out, states, audio_len

def convert_idx_to_token_tensor(inputs, id2char):
    """Converts int32 tensor to string tensor.
    Reference:
        https://github.com/Kyubyong/transformer/blob/master/utils.py
    """
    def my_func(inputs):
        sent = "".join(id2char[elem] for elem in inputs)
        sent = sent.split("<EOS>")[0].strip()
        sent = sent.replace("<SPACE>", " ")
        return " ".join(sent.split())

    return tf.py_func(my_func, [inputs], tf.string)

def convert_idx_to_string(inputs, id2char):
    """Converts int32 ndarray to string."""

    sent =  "".join(id2char[elem] for elem in inputs)
    sent = sent.split("<EOS>")[0].strip()
    sent = sent.replace("<SPACE>", " ") 
    return " ".join(sent.split())

def get_texts(y_hat, sess, num_batches, id2char):
    output_id = []
    output_char = []
    for _ in range(num_batches):
        pred = sess.run(y_hat)
        output_id += pred.tolist()
    for h in output_id:
        sent = "".join(id2char[idx] for idx in h)
        sent = sent.split("<EOS>")[0].strip()
        sent = sent.replace("<SPACE>", " ") 
        output_char.append(" ".join(sent.split()))

    return output_char
