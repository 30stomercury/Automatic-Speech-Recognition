from tensor2tensor.layers import common_audio
import tensorflow as tf
import librosa
import speechpy
import numpy as np
from glob import glob
import string
import os
from tqdm import tqdm
from las.arguments import *

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

def CMVN(audios):
    # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/speech_recognition.py
    # per utterance normalization
    var_epsilon = 1e-09
    mean = tf.reduce_mean(audios, keepdims=True, axis=1)
    variance = tf.reduce_mean(tf.squared_difference(audios, mean),
                              keepdims=True, axis=1)
    audios = (audios - mean) * tf.rsqrt(variance + var_epsilon)
    return audios

def process_audios_batch(audio_path, 
                         sess, 
                         prepro_batch=128, 
                         sample_rate=22050,
                         frame_step=10, 
                         frame_length=25,
                         feat_dim = 40,
                         feat_type='fbank',
                         dither=0,
                         cmvn=True):
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
               each element is in the shape of (L, feat_dim).
        featlen: List of feature length.
    """    

    if feat_type != "fbank":
        raise NotImplementedError(
                "Only support fbank.")

    # build extacting graph
    input_audio = tf.placeholder(dtype=tf.float32, shape=[None, None])
    if feat_type == 'fbank':
        mel_fbanks = common_audio.compute_mel_filterbank_features(
            input_audio, sample_rate=sample_rate, dither=dither, frame_step=frame_step, frame_length=frame_length, num_mel_bins=feat_dim, apply_mask=True)
        mel_fbanks = tf.reduce_sum(mel_fbanks, -1)
    if cmvn:
        mel_fbanks = CMVN(mel_fbanks)

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

def process_audios(audio_path,
                   frame_step=10, 
                   frame_length=25,
                   feat_dim=40,
                   feat_type='fbank',
                   cmvn=True):
    """
    Returns:
        feats: List of features with variable length L, 
               each element is in the shape of (L, 39), 13 for mfcc,
               26 for its firs & second derivative.
        featlen: List of feature length.
    """    
    feats = []
    featlen = []
    for p in tqdm(audio_path):

        audio, fs = librosa.load(p)

        if feat_type == 'mfcc':
            assert feat_dim == 13, "13 is commonly used"
            mfcc = speechpy.feature.mfcc(audio, fs, frame_length=frame_length/1000, frame_stride=frame_step/1000, num_cepstral=feat_dim)
            if cmvn:
                mfcc = speechpy.processing.cmvn(mfcc, True)
            mfcc_39 = speechpy.feature.extract_derivative_feature(mfcc)
            feats.append(mfcc_39.reshape(-1, feat_dim*3).astype(np.float32))

        elif feat_type == 'fbank':
            fbank = speechpy.feature.lmfe(audio, fs, frame_length=frame_length/1000, frame_stride=frame_step/1000, num_filters=feat_dim)
            if cmvn:
                fbank = speechpy.processing.cmvn(fbank, True)
            feats.append(fbank.reshape(-1, feat_dim).astype(np.float32))

        featlen.append(len(feats))

    return feats, np.array(featlen).astype(np.int32)

def process_texts(special_chars, texts):
    """
    Returns:
        chars: List of index sequences.
        charlen: List of length of sequences.
    """

    charlen = []
    chars = []
    char2id, id2char = lookup_dicts(special_chars)
    for sentence in tqdm(texts):
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        char_converted = [char2id[char] if char != ' ' else char2id['<SPACE>'] for char in list(sentence)]
        chars.append(char_converted + [char2id['<EOS>']])
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


def main():
    # arguments
    args = parse_args()

    # define data generator
    train_libri_path = args.train_data_path
    dev_libri_path = args.dev_data_path
    train_texts, train_audio_path = data_preparation(train_libri_path)
    dev_texts, dev_audio_path = data_preparation(dev_libri_path)

    print("Process train/dev features...")
    if not os.path.exists(args.feat_path):
        os.makedirs(args.feat_path)

    # texts
    special_chars = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
    train_chars, train_charlen, char2id, id2char = process_texts(special_chars, train_texts)
    dev_chars, dev_charlen, _, _ = process_texts(special_chars, dev_texts)
    np.save(args.feat_path+"/train_chars.npy", train_chars)
    np.save(args.feat_path+"/train_charlen.npy", train_charlen)
    np.save(args.feat_path+"/dev_chars.npy", dev_chars)
    np.save(args.feat_path+"/dev_charlen.npy", dev_charlen)

    # audios
    print("Process train audios...")
    train_feats, train_featlen = process_audios(train_audio_path,
                                                frame_step=10, 
                                                frame_length=25,
                                                feat_dim=40,
                                                feat_type='fbank',
                                                cmvn=True)
    np.save(args.feat_path+"/train_feats.npy", train_feats)    
    np.save(args.feat_path+"/train_featlen.npy", train_featlen)

    print("Process dev audios...")
    dev_feats, dev_featlen = process_audios(dev_audio_path,
                                            frame_step=10, 
                                            frame_length=25,
                                            feat_dim=40,
                                            feat_type='fbank',
                                            cmvn=True)
    np.save(args.feat_path+"/dev_feats.npy", dev_feats)
    np.save(args.feat_path+"/dev_featlen.npy", dev_featlen)

if __name__ == '__main__':
    main()
