from tensor2tensor.layers import common_audio
import tensorflow as tf
import librosa
import numpy as np
from glob import glob
import re
import time

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
            texts.append(line[len(line_[0])+1:])
    return texts, audio_path

def load_audio(audio_path, 
               sess, 
               prepro_batch=128, 
               sample_rate=22050,
               frame_step=10, 
               feat_dim = 40,
               feat_type='fbank'):
    """GPU accerated audio features extracting in tensorflow

    Args:
        audio_path: Path audio files.
        sess: Tf session to execute the graph for feature extraction.
        prepro_batch: Batch size for preprocessing audio features.
        frame_step: Step size in ms.
        feat_dim: Feature dimension.
        feat_type: Types of features you want to apply.

    Returns:
        feats_list: List of features with variable length L, 
                    each element is in the shape of (L, feat_dim), N is
                    the number of samples.
        length_list1: List of feature length.
    """    

    # build extacting graph
    input_audio = tf.placeholder(dtype=tf.float32, shape=[None, None])
    if feat_type == 'fbank':
        mel_fbanks = common_audio.compute_mel_filterbank_features(
            input_audio, sample_rate=sample_rate, frame_step=frame_step, num_mel_bins=feat_dim, apply_mask=True)
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
    feats_list = []
    length_list = []

    # start extracting audio feature in a batch manner:
    for p in audio_path:
        audio, fs = librosa.load(p)
        audio_batch.append(audio)
        len_batch.append(len(audio))
        if len(audio_batch) == prepro_batch:
            feat, feat_len = extract_feat(audio_batch, len_batch, fs)
            # remove paddings of audios batch:
            for index, l in enumerate(feat_len):
                feats_list.append(feat[index][:l])
            length_list = np.concatenate([length_list, feat_len])
            audio_batch = []
            len_batch = []
            print("Processed samples: {}/{}".format(len(feats_list), len(audio_path)))

    if len(audio_batch) % prepro_batch != 0:
        feat, feat_len = extract_feat(audio_batch, len_batch, fs)
        # remove paddings:
        for index, l in enumerate(feat_len):
            feats_list.append(feat[index][:l])
        length_list = np.concatenate([length_list, feat_len])
        print("Processed samples: {}/{}".format(len(feats_list), len(audio_path)))

    return feats_list, length_list.astype(np.int32)


def batch_gen(feats_list, length_list1, batch_size=5):

    """
    Returns:
        batches        

    """
    assert len(feats_list) == len(length_list1)
    num_batches = len(feats_list) // batch_size + int(len(feats_list) % batch_size != 0)
    def generator():
        buff = []
        for i, x in enumerate(feats_list):
            if i % batch_size == 0:
                len_batch = length_list1[i:i+batch_size]
                max_len = max(len_batch)
            
            if i % batch_size == 0 and buff:
                yield np.stack(buff, 0), len_batch
                buff = []
            padded = np.zeros([max_len - x.shape[0], x.shape[1]], dtype=np.float32)
            x_padded = np.concatenate([x, padded], 0)
            buff.append(x_padded)

        if buff:
            yield np.stack(buff, 0), len_batch

    shapes = ([None, None, None], [None])
    types = (tf.float32, tf.int32)
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types=types, 
                                             output_shapes=shapes)
    dataset.batch(batch_size).prefetch(1)
    dataset = dataset.shuffle(128*batch_size)
    dataset = dataset.repeat() 
    iter = dataset.make_initializable_iterator()
    return iter, num_batches

#def load_text(path):

if __name__ == "__main__":
    s = time.time()
    sess = tf.Session()
    libri_path = './data/LibriSpeech/dev-clean'
    texts, audio_path = data_preparation(libri_path)
    X, X_len = load_audio(audio_path, sess, prepro_batch=64)
    iter_, num_batches = batch_gen(X, X_len, batch_size=32)
    x = iter_.get_next()
    sess.run(tf.global_variables_initializer())
    sess.run(iter_.initializer)
    print(num_batches)
    for _ in range(100):
        print(sess.run(x)[0].shape, max(sess.run(x)[1]), sess.run(x)[1].shape)
    print(time.time()-s)
