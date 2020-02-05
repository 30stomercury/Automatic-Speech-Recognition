import tensorflow as tf
import numpy as np

def batch_gen(feats, chars, featlen, charlen, batch_size, feat_dim,  bucketing=True, is_training=True):
    """
    Returns:
        iter: Batch iterator.
        batch_num: Number of batches.        
    """

    # Check if the number of sample points matches.
    assert len(feats) == len(chars)
    assert len(feats) == len(featlen)
    assert len(chars) == len(charlen)
    num_batches = len(feats) // batch_size + int(len(feats) % batch_size != 0)

    def generator():
        buff_feats = []
        buff_chars = []

        if not bucketing:
            rand_idx = np.random.permutation(len(feats))
            feats_, featlen_  = feats[rand_idx], featlen[rand_idx]
            chars_, charlen_  = chars[rand_idx], charlen[rand_idx]
        else:
            sort_idx = featlen.argsort()
            if is_training:
                window = 4000
                for i in range(len(sort_idx) // window):
                    sort_idx[i*window:(i+1)*window] = sort_idx[i*window:(i+1)*window][np.random.permutation(window)]
            feats_, featlen_ = feats[sort_idx], featlen[sort_idx]
            chars_, charlen_ = chars[sort_idx], charlen[sort_idx]
            
        for i, x in enumerate(zip(feats_, chars_)):
            if i % batch_size == 0 and buff_feats and buff_chars:
                yield (np.stack(buff_feats, 0), len_batch1), (np.stack(buff_chars, 0), len_batch2)
                buff_feats = []
                buff_chars = []
            if i % batch_size == 0:
                len_batch1 = featlen_[i:i+batch_size]
                len_batch2 = charlen_[i:i+batch_size]
                max_len1 = max(len_batch1)
                max_len2 = max(len_batch2)
            # Padding
            x_feat, x_char = x
            padded_feat = np.zeros([max_len1 - x_feat.shape[0], x_feat.shape[1]], dtype=np.float32)
            padded_char = np.zeros(max_len2 - len(x_char), dtype=np.int32)
            feat_padded = np.concatenate([x_feat, padded_feat], 0)
            char_padded = np.concatenate([x_char, padded_char], 0)
            buff_feats.append(feat_padded)
            buff_chars.append(char_padded)

        if buff_feats and buff_chars:
            yield (np.stack(buff_feats, 0), len_batch1), (np.stack(buff_chars, 0), len_batch2)

    shapes = (([None, None, feat_dim], [None]), ([None, None], [None]))
    types = ((tf.float32, tf.int32), (tf.int32, tf.int32))
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types=types, 
                                             output_shapes=shapes)
    dataset.batch(batch_size).prefetch(1)
    if is_training:
        dataset = dataset.shuffle(batch_size*64)
    dataset = dataset.repeat() 
    iter = dataset.make_initializable_iterator()
    return iter, num_batches


