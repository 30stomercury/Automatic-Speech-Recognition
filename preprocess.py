import tensorflow as tf
import librosa
import speechpy
import numpy as np
from glob import glob
import string
import os
import sox
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
            texts.append(line[len(line_[0])+1:-1].replace("'",""))
    return texts, audio_path

def speed_augmentation(filelist, target_folder, speed_list):
    """Speed Augmentation
    
    Args:
        path: Path to audio files.
        target_folder: Folder of augmented audios.
    """
    audio_path = []
    aug_generator = sox.Transformer()
    print("Total audios:", len(filelist))
    for speed in speed_list:
        aug_generator.speed(speed)
        target_folder_ = target_folder+"_"+str(speed)
        if not os.path.exists(target_folder_):
            os.makedirs(target_folder_)
        for source_filename in tqdm(filelist):
            file_id = source_filename.split("/")[-1]
            save_filename = target_folder_+"/"+file_id.split(".")[0]+"_"+str(speed)+"."+file_id.split(".")[1] 
            aug_generator.build(source_filename, save_filename)
            audio_path.append(save_filename)
    return audio_path

def CMVN(audios):
    # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/speech_recognition.py
    # per utterance normalization
    var_epsilon = 1e-09
    mean = tf.reduce_mean(audios, keepdims=True, axis=1)
    variance = tf.reduce_mean(tf.squared_difference(audios, mean),
                              keepdims=True, axis=1)
    audios = (audios - mean) * tf.rsqrt(variance + var_epsilon)
    return audios


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

        try:
            audio, fs = librosa.load(p)
        except:
            print(p)
            pass

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

        featlen.append(len(feats[-1]))

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
    
    """
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
                                                feat_dim=13,
                                                feat_type='mfcc',
                                                cmvn=True)
    np.save(args.feat_path+"/train_feats.npy", train_feats)    
    np.save(args.feat_path+"/train_featlen.npy", train_featlen)

    print("Process dev audios...")
    dev_feats, dev_featlen = process_audios(dev_audio_path,
                                            frame_step=10, 
                                            frame_length=25,
                                            feat_dim=13,
                                            feat_type='mfcc',
                                            cmvn=True)
    np.save(args.feat_path+"/dev_feats.npy", dev_feats)
    np.save(args.feat_path+"/dev_featlen.npy", dev_featlen)
    """

    # augmentation
    if args.augmentation:
        aug_audio_path = speed_augmentation(filelist=train_audio_path,
                                            target_folder="data/LibriSpeech/LibriSpeech_aug", 
                                            speed_list=[1.1])
        aug_feats, aug_featlen = process_audios(aug_audio_path,
                                                    frame_step=10, 
                                                    frame_length=25,
                                                    feat_dim=13,
                                                    feat_type='mfcc',
                                                    cmvn=True)
        np.save(args.feat_path+"/aug_feats1.2.npy", aug_feats)    
        np.save(args.feat_path+"/aug_featlen1.2.npy", aug_featlen)

if __name__ == '__main__':
    main()
