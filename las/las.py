import numpy as np
import tensorflow as tf

from las import utils

class Listener:

    def __init__(self, args):
        self.args = args

    def __call__(self, inputs):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            enc_out, enc_states = pblstm(inputs, self.args.num_enc_layers, self.args.enc_units, self.args.keep_proba, self.args.is_training)
        return enc_out, enc_state

class Speller:

    def __init__(self, args):
        self.args = args
        self._build_decoder_cell()
        self._build_char_embeddings()

    def __call__(self, enc_out, dec_step, teacher):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            prev_char = tf.nn.embedding_lookup(
                            self.embedding_matrix, tf.zeros(self.args.batch_size))
            dec_state = dec_cell.zero_state(self.args.batch_size, tf.float32)
            output = []
            alphas = []
            for t in range(dec_steps):
                context = self.attention(h=enc_out, 
                                         char=prev_char, 
                                         hidden_size=self.args.embedding_size, 
                                         seqlength=seqlength)
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

    def __init__(self, args, listener, speller):
        '''Consturct Listen attend ande spell objects.
        Args:
            args: Include model/train/inference parameters that are packed in arguments.py
        '''
        self.args = args
        self.listener = listener(args)
        self.speller = speller(args)

    def train(self, x, y):
        ''' Buid decoder encoder network, compute loss.
        Args:
            x: (B, T1, D), T1: feature timesteps.
            y: (B, T2), T2: output time steps.
        Returns: 
            loss: scalar.
            train_op: training operation.
            global_step: scalar.
            summaries: tf-board summarise
        '''
        # encoder decoder network
        h, enc_state = self.listener(x)
        output_seq = self.speller()
        
        
        




