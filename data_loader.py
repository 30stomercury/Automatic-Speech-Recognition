from tensor2tensor.layers import common_audio
import tensorflow as tf
import librosa
import numpy as np
from glob import glob
import string

def data_preparation(libri_path):
    """Prepare texts and its corresponding audio file path
    
    Args:
        path: Path to texts and audio files.
    
    Returns:
        texts: List of sentences.
        audio_path: Audio paths of its corresponding sentences.
    """

    folders = glob(libri_path+"/**/**")
    texts = []
    audio_path = []
    for path in folders:
        text_path = glob(path+"/*txt")[0]
        f = open(text_path)
        for line in f.readlines():
            line_ = line.split(" ")
            audio_path.append(path+"/"+line_[0]+".flac")
            texts.append(line[len(line_[0])+1:-1])
    return texts, audio_path

def process_audio(audio_path, 
                  sess, 
                  prepro_batch=128, 
                  sample_rate=22050,
                  frame_step=10, 
                  frame_length=25,
                  feat_dim = 40,
                  feat_type='fbank'):
    """GPU accerated audio features extracting in tensorflow

    Args:
        audio_path: List of path of audio files.
        sess: Tf session to execute the graph for feature extraction.
        prepro_batch: Batch size for preprocessing audio features.
        frame_step: Step size in ms.
        feat_dim: Feature dimension.
        feat_type: Types of features you want to apply.

    Returns:
        feats: List of features with variable length L, 
               each element is in the shape of (L, feat_dim), N is
               the number of samples.
        featlen: List of feature length.
    """    

    # build extacting graph
    input_audio = tf.placeholder(dtype=tf.float32, shape=[None, None])
    if feat_type == 'fbank':
        mel_fbanks = common_audio.compute_mel_filterbank_features(
            input_audio, sample_rate=sample_rate, frame_step=frame_step, frame_length=frame_length, num_mel_bins=feat_dim, apply_mask=True)
        mel_fbanks = tf.reduce_sum(mel_fbanks, -1)

    def extract_feat(audio_batch, len_batch, fs):
        max_len = max(len_batch)
        audio_padded = np.zeros([prepro_batch, max_len], dtype=np.float32)
        for i in range(len(audio_batch)):
            audio_padded[i][:len(audio_batch[i])] = audio_batch[i]
        feat = sess.run(mel_fbanks, feed_dict={input_audio: audio_padded})
        # compute the feature length:
        feat_len = np.array(len_batch) // int(fs * frame_step / 1e3) + 1
        feat_len = feat_len.astype(np.int32)
        return feat, feat_len
        
    audio_batch = []
    len_batch = []
    feats = []
    featlen = []

    # start extracting audio feature in a batch manner:
    for p in audio_path:
        audio, fs = librosa.load(p)
        audio_batch.append(audio)
        len_batch.append(len(audio))
        if len(audio_batch) == prepro_batch:
            feat, feat_len = extract_feat(audio_batch, len_batch, fs)
            # remove paddings of audios batch:
            for index, l in enumerate(feat_len):
                feats.append(feat[index][:l])
            featlen = np.concatenate([featlen, feat_len])
            audio_batch = []
            len_batch = []
            print("Processed samples: {}/{}".format(len(feats), len(audio_path)))

    if len(audio_batch) % prepro_batch != 0:
        feat, feat_len = extract_feat(audio_batch, len_batch, fs)
        # remove paddings:
        for index, l in enumerate(feat_len):
            feats.append(feat[index][:l])
        featlen = np.concatenate([featlen, feat_len])
        print("Processed samples: {}/{}".format(len(feats), len(audio_path)))

    return np.array(feats), featlen.astype(np.int32)

def process_texts(special_chars, texts):
    """
    Returns:
        chars: List of index sequences.
        charlen: List of length of sequences.
    """

    charlen = []
    chars = []
    char2id, id2char = lookup_dicts(special_chars)
    for sentence in texts:
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        char_converted = [char2id[char] if char != ' ' else char2id['<SPACE>'] for char in list(sentence)]
        chars.append([char2id['<SOS>']] + char_converted + [char2id['<EOS>']])
        charlen.append(len(chars[-1]))

    return np.array(chars), np.array(charlen).astype(np.int32), char2id, id2char

def lookup_dicts(special_chars):
    """
    Args:
        special_chars: special charactors, <PAD>, <SOS>, <EOS>, <SPACE>
    Returns:
        char2id: dict, from character to index.
        id2char: dict, from index to character.
    """

    alphas = list(string.ascii_uppercase[:26])
    chars = special_chars + alphas
    char2id = {}
    id2char = {}
    for i, c in enumerate(chars):
        char2id[c] = i
        id2char[i] = c
    return char2id, id2char

def batch_gen(feats, chars, featlen, charlen, batch_size, feat_dim,  bucketing=True, shuffle_batches=True):
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
    if shuffle_batches:
        dataset = dataset.shuffle(batch_size*64)
    dataset = dataset.repeat() 
    iter = dataset.make_initializable_iterator()
    return iter, num_batches

if __name__ == "__main__":
    sess = tf.Session()
    libri_path = './data/LibriSpeech/dev-clean'
    texts, audio_path = data_preparation(libri_path)
    special_chars = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
    chars, charlen = process_texts(special_chars, texts)
    X, X_len = process_audio(audio_path, sess, prepro_batch=64)
    iter_, num_batches = batch_gen(X, chars, X_len, charlen, batch_size=32)
    x = iter_.get_next()
    sess.run(tf.global_variables_initializer())
    sess.run(iter_.initializer)
    print(num_batches)
    for _ in range(5):
        a, b = sess.run(x)
        print(a[0].shape, a[1].shape, b[0].shape, b[1].shape)
