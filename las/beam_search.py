import numpy as np
import tensorflow as tf

class BeamState(object):

    def __init__(self, char_ids, log_prob, state):
        """hypothesis
        Args:
            char_id: Index of the starting charactor, 1 corresponding to <SOS>.       
            log_prob: Log probability of the starting charactor.
            state: Decoder initial state. Zero state of RNN cell.
        """
        self.char_ids = char_ids
        self.log_prob = log_prob
        self.state = state

    def update(self, char_id, log_prob, state):
        """Return new beam state based on last decoding results."""
        return BeamState(self.char_ids+[char_id], self.log_prob+log_prob, state)
    
class BeamSearch(object):

    def __init__(self, args, las, char2id):
        """Beam search decoder
        args: arguments.
        las: LAS.
        char2id: dict, map from charactor to index.
        beam_size: int, beam size.
        """    
        self.args = args
        self.listener = las.listener
        self.speller = las.speller
        self.beam_size = args.beam_size
        self.char2id = char2id
        self.start_id = char2id['<SOS>']
        self.end_id = char2id['<EOS>']
        # build graph
        self._build_init(size=1)
        self._build_encode()
        self._build_decode_step()
        print("Graph built")

    def decode(self, sess, xs):
        """
        sess: tf session
        xs: a tuple of 
            - audio:    (1, T1, D), T1: padded feature timesteps.
            - audiolen: (1,), original feature length.
        """
        audio, audiolen = xs
        # beam search can only perform on single utterance.
        if len(audio) != 1:
          raise ValueError('batch size must be 1 while performing beam search.')
        
        h, enc_len = self._get_encode(sess, audio, audiolen)
        # estimate decoding steps
        dec_step = int(audiolen * self.args.convert_rate)
        t = 0
        # decode
        init_state = self._get_init(sess)
        beam_set = [BeamState([self.start_id], 0, init_state)] #* self.beam_size
        selected_beam_state = []
        while t < dec_step and len(selected_beam_state) < self.beam_size:
            prev_char_ids = [b.char_ids[-1] for b in beam_set]
            dec_states = [b.state for b in beam_set]
            beam_set_bank = []
         
            # collect search path from beam_size to 2*beam_size*beam_size
            for i in range(len(beam_set)):
                logits, dec_state = self._get_decode(sess, h, enc_len, prev_char_ids[i], dec_states[i])
                topk_ids = np.argsort(logits)       # => argsort is in acending order
                topk_probs = logits[topk_ids]
                for j in range(self.args.vocab_size):
                    beam_set_bank.append(beam_set[i].update(topk_ids[j], topk_probs[j], dec_state))
            beam_set = []                 
            # sort by log prob
            topk_beam_state = self._select_best_k(beam_set_bank)
            for b in topk_beam_state:
                if b.char_ids[-1] == self.end_id:
                    selected_beam_state.append(b)
                else:
                    beam_set.append(b)
            t += 1
        
        return self._select_best_k(selected_beam_state)

    def _build_init(self, size):
        """get decoder initial state"""
        self.init_state = self.speller.dec_cell.zero_state(size, tf.float32)

    def _build_encode(self):
        """build encoder graph"""
        # for encoder
        self.audio = tf.placeholder(tf.float32,
                                     shape=[None, None, self.args.feat_dim], name='audio')
        self.audiolen = tf.placeholder(tf.int32, shape=[None], name='audiolen')
        # build graph
        self.enc_out, enc_state, self.enc_len = self.listener(self.audio, self.audiolen, is_training=False)

    def _build_decode_step(self):
        """build decoder graph"""
        # for decoder
        self.h = tf.placeholder(tf.float32,
                                      shape=[None, None, self.args.enc_units*2], name='enc_out')
        self.h_len = tf.placeholder(tf.int32, shape=[None], name='enc_len')
        self.prev_char_id = tf.placeholder(tf.int32, name='char_id')
        self.rnn_state_packed = tf.placeholder(tf.float32, 
                                    [self.args.num_dec_layers, self.args.batch_size, self.args.dec_units], name='rnn_state')
        # form to tuple state
        l = tf.unstack(self.rnn_state_packed, axis=0)
        rnn_tuple_state = tuple(
                [l[i]
                 for i in range(self.args.num_dec_layers)])
        # look up
        prev_char = self.speller.look_up(self.prev_char_id)
        prev_char = tf.reshape(prev_char, [1, self.args.embedding_size])
        # build graph, specify the variable scope
        self.cur_char, self.rnn_state, alphas = self.speller.decode(
                            self.h, self.h_len, rnn_tuple_state, prev_char, is_training=False)

    def _get_encode(self, sess, audio, audiolen):
        """get encoder output"""
        feed_dict = {
                    self.audio: audio,
                    self.audiolen: audiolen
                    }
        return sess.run([self.enc_out, self.enc_len], feed_dict)

    def _get_decode(self, sess, enc_out, enc_len, prev_char_id, rnn_state_packed):
        """build decoder output"""
        feed_dict = {
                    self.h: enc_out,
                    self.h_len: enc_len,
                    self.prev_char_id: prev_char_id,
                    self.rnn_state_packed: rnn_state_packed
                    }
        logits, state = sess.run([self.cur_char, self.rnn_state], feed_dict)
        return logits[0], state

    def _get_init(self, sess):
        """get rnn init state"""
        return sess.run(self.init_state)

    def _select_best_k(self, beam_set):
        """select top k BeamStatei
        Args:
            beam_set: list, beam set.
        Returns:
            beam_set: list, sorted beam_set based on log_prob in acending order.
        """
        log_prob = [b.log_prob for b in beam_set]
        idx = np.argsort(log_prob)[-self.beam_size:]
        return [beam_set[i] for i in idx]
        

        





