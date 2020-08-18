import numpy as np
import tensorflow as tf

NORM = True


class BeamState(object):

    def __init__(self, token_ids, log_prob, att, dec_state, lm_state):
        """hypothesis
        Args:
            token_id: List, Index of the starting tokenactor, 1 corresponding to <SOS>.       
            log_prob: Float, Log probability of the starting tokenactor.
            att:      List, attention weights.
            state:    Tuple, Decoder initial state. Zero state of RNN cell.
        """
        self.token_ids = token_ids
        self.log_prob = log_prob
        self.att = att
        self.dec_state = dec_state
        self.lm_state = lm_state

    def update(self, token_id, log_prob, att, dec_state, lm_state):
        """Return new beam state based on last decoding results."""
        return BeamState(
                    self.token_ids+[token_id],
                    self.log_prob+log_prob, 
                    self.att + [att],
                    dec_state, 
                    lm_state)
    
class BeamSearch(object):

    def __init__(self, args, las, token_to_id, language_model):
        """Beam search decoder
        Args:
            args: arguments.
            las: LAS.
            token_to_id: dict, map from token to index.
            language_model: class, RNNLM.
        """    
        self.args = args
        self.listener = las.listener
        self.speller = las.speller
        self.beam_size = args.beam_size
        self.token_to_id = token_to_id
        self.start_id = token_to_id['<SOS>']
        self.end_id = token_to_id['<EOS>']

        if args.unit.lower() != "char" and args.unit.lower() != "subword":
          raise ValueError('Other units are currently not support!')

        # build graph
        self._build_encode(self.args.enc_type)
        self._build_decode()
        
        # Language Model
        if args.apply_lm:
            self.lm = language_model

    def decode(self, sess, xs): # TODO rewrite decode -> clean codes
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
        
        # encode        
        h, enc_len = self._get_encode(sess, audio, audiolen)

        # estimate decoding steps
        dec_step = int(audiolen * self.args.convert_rate)
        t = 0

        # decode
        dec_init_state, lm_init_state = self._get_dec_init(sess), None

        if self.args.apply_lm:
            lm_init_state = self._get_lm_init(sess)

        beam_set = [BeamState(token_ids=[self.start_id], 
                              log_prob=0, 
                              att=[np.zeros(h.shape[1])], 
                              dec_state=dec_init_state, 
                              lm_state=lm_init_state)] * self.beam_size 
        selected_beam_state = []

        while t < dec_step and len(selected_beam_state) < self.beam_size:
            prev_token_ids = [b.token_ids[-1] for b in beam_set]
            prev_align = [b.att[-1] for b in beam_set]

            ## used to collect beams
            beam_set_bank = []

            # put N nodes into a batch and feed it to the model
            prev_dec_states = [self._pack_state(b.dec_state) for b in beam_set]
            prev_dec_states = np.concatenate(prev_dec_states, 1)

            # decoding
            logits, dec_states, alphas = self._get_decode(
                                    sess, h, enc_len, prev_token_ids, prev_align, prev_dec_states)

            if self.args.apply_lm: 
                prev_lm_states = [b.lm_state for b in beam_set]
                prev_lm_states = np.split(
                                np.concatenate(prev_lm_states, 2).reshape([-1, 512]), 4) # TODO move to args
                # resoring w/ language model
                lm_out, lm_states = self._get_lm(
                                        sess, prev_token_ids, prev_lm_states)
                logits[:, 2:] += lm_out*self.args.lm_weight


            num_beam = len(beam_set) if t > 0 else 1 

            # collect nodes
            for i in range(num_beam):
                topk_ids = np.argsort(logits[i])[-64:]      # => argsort is in acending order
                topk_probs = logits[i][topk_ids]

                for j in range(len(topk_ids)):              # => search each node
                    if t > 0 and topk_ids[j] == self.start_id:
                        continue

                    if self.args.apply_lm:
                        beam_set_bank.append(beam_set[i].update(topk_ids[j], 
                                                        topk_probs[j], 
                                                        alphas[i]
                                                        (dec_states[0][i], dec_states[1][i]),
                                                        lm_states[:, :, i, :]))
                    else:
                        beam_set_bank.append(beam_set[i].update(topk_ids[j], 
                                                        topk_probs[j], 
                                                        alphas[i],
                                                        (dec_states[0][i], dec_states[1][i]),
                                                        None))

            # used to collect topk beams
            beam_set = []                 

            # sort by log prob
            topk_beam_state = self._select_best_k(beam_set_bank, NORM)
            for b in topk_beam_state:
                if b.token_ids[-1] == self.end_id:
                    selected_beam_state.append(b)
                else:
                    beam_set.append(b)
            t += 1
        
        if t == dec_step:
            selected_beam_state.extend(beam_set)

        return self._select_best_k(selected_beam_state, NORM)

    def _build_encode(self, enc_type):
        """build encoder graph"""
        # for encoder
        self.audio = tf.placeholder(tf.float32,
                                     shape=[None, None, self.args.feat_dim, 3], name='audio')
        self.audiolen = tf.placeholder(tf.int32, shape=[None], name='audiolen')

        # build graph
        self.enc_out, enc_state, self.enc_len = \
                        self.listener(self.audio, self.audiolen, encoder=enc_type, is_training=False)

    def _build_decode(self):
        """build decoder graph"""
        # for decoder
        self.h = tf.placeholder(tf.float32,
                                shape=[None, None, self.args.enc_units], 
                                name='enc_out')
        self.h_len = tf.placeholder(tf.int32, shape=[None], name='enc_len')
        self.prev_token_id = tf.placeholder(tf.int32, shape=[None], name='token_id')
        self.prev_align = tf.placeholder(tf.float32,
                                shape=[None, None])
        self.dec_state_packed = tf.placeholder(tf.float32, 
                                [self.args.num_dec_layers, None, self.args.dec_units], 
                                name='rnn_state')

        # form to tuple state
        l = tf.unstack(self.dec_state_packed, axis=0)
        rnn_tuple_state = tuple(
                [l[i]
                 for i in range(self.args.num_dec_layers)])

        # look up
        prev_token = self.speller._look_up(self.prev_token_id)
        prev_token = tf.reshape(prev_token, [-1, self.args.embedding_size])

        # build graph, specify the variable scope
        self.cur_token, self.dec_state, self.alphas = self.speller.decode(self.h, 
                                                            self.h_len, 
                                                            rnn_tuple_state, 
                                                            prev_token, 
                                                            self.prev_align, 
                                                            is_training=False)

    def _get_encode(self, sess, audio, audiolen):
        """get encoder output"""
        feed_dict = {
                    self.audio: audio,
                    self.audiolen: audiolen
                    }
        return sess.run([self.enc_out, self.enc_len], feed_dict)

    def _get_decode(
            self, sess, enc_out, enc_len, prev_token_id, prev_align, dec_state_packed):
        """get decoder output"""
        N = len(prev_token_id)
        feed_dict = {
                    self.h: np.tile(enc_out, (N, 1, 1)),
                    self.h_len: np.tile(enc_len, N),
                    self.prev_token_id: prev_token_id,
                    self.prev_align: prev_align,
                    self.dec_state_packed: dec_state_packed
                    }
        logits, states, alphas = sess.run([self.cur_token, self.dec_state, self.alphas], feed_dict)

        return logits, states, alphas
    
    def _get_lm(self, sess, prev_token_id, lm_state_packed):
        """get decoder output"""
        shifted_id = [i-2 for i in prev_token_id]
        shifted_id =  np.reshape(shifted_id, [-1, 1])

        lm_state, logits = sess.run([self.lm.final_state,
                                   self.lm.logits],
                                  {self.lm.input_data: shifted_id,
                                   self.lm.initial_state: lm_state_packed})

        return logits, np.array(lm_state)

    def _get_dec_init(self, sess):
        """get decoder init state"""
        return sess.run(self.speller.dec_cell.zero_state(1, tf.float32))

    def _get_lm_init(self, sess):
        """get decoder init state"""
        lm_init_state = sess.run(self.lm.zero_state)

        return np.array(lm_init_state) # convert LSTMTuple into nd array

    def _pack_state(self, state):
        """Packed state to put N nodes into a batch"""
        return (state[0].reshape(1,-1), state[1].reshape(1,-1))

    def _get_decode_varlist(self, save_path):
        """Get variables for decode graph."""
        # create restore dict for decode scope
        var = {}
        var_all = tf.global_variables()
        var_att = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'decode/attention/Variable')
        var_decode = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'decode/')
        var_list = [i[0] for i in tf.train.list_variables(save_path)]

        for v in var_all:
            name = v.name.split(":")[0]
            if v in var_att:
                var['Speller/while/' + name] = v
            elif v in var_decode and v not in var_att:
                var['Speller/' + name] = v
            elif name in var_list and name != "global_step":
                var[v.name.split(":")[0]] = v

        return var

    def restore_las(self, sess, save_path, restore_epoch):
        """Restore LAS ckpt."""
        ckpt = tf.train.latest_checkpoint(save_path)
        if restore_epoch != -1:
            ckpt = save_path+"las_E{}".format(restore_epoch)
        var_las = self._get_decode_varlist(ckpt) 
        # restore
        saver = tf.train.Saver(var_list=var_las)
        saver.restore(sess, ckpt)
        return ckpt

    def restore_lm(self, sess, save_path):
        """Restore LM ckpt."""
        with tf.name_scope('evaluation'):
            var_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '')
            var_list = [i[0] for i in tf.train.list_variables(save_path)]
            var_lm = {}
            for v in var_all:
                if v.name.split(":")[0] in var_list:
                    var_lm[v.name.split(":")[0]] = v

            # create restore dict for decode scope
            saver_lm = tf.train.Saver(name='checkpoint_saver', var_list=var_lm)
            saver_lm.restore(sess, save_path)

    def _select_best_k(self, beam_set, norm=False):
        """select top k BeamState
        Args:
            beam_set: list, beam set.
            norm: bool, normalize with length.
        Returns:
            beam_set: list, sorted beam_set based on log_prob in acending order.
        """
        if norm:
            log_prob = [b.log_prob/(len(b.token_ids)-1) for b in beam_set]
        else:
            log_prob = [b.log_prob for b in beam_set]

        idx = np.argsort(log_prob)[-self.beam_size:]

        return [beam_set[i] for i in idx]
        

        





