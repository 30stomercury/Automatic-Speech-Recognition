import tensorflow as tf
import librosa
import speechpy
import numpy as np
from glob import glob
import string
import os
from tqdm import tqdm
from tokenizers import CharBPETokenizer
from las.arguments import *
from utils.text import text_encoder
from utils.augmentation import speed_augmentation, volume_augmentation

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

def process_audios(audio_path, args):
    """
    Returns:
        feats: List of features with variable length L, 
               each element is in the shape of (L, 39), 13 for mfcc,
               26 for its firs & second derivative.
        featlen: List of feature length.
    """   
    # setting 
    frame_step = args.frame_step
    frame_length = args.frame_length
    feat_dim = args.feat_dim
    feat_type = args.feat_type
    cmvn = args.cmvn
    # run
    feats = []
    featlen = []
    for p in tqdm(audio_path):

        try:
            audio, fs = librosa.load(p, None)
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

def process_texts(special_tokens, texts, tokenizer):
    """
    Returns:
        tokens: List of index sequences.
        tokenlen: List of length of sequences.
    """
    tokenlen = []
    tokens = []
    for sentence in tqdm(texts):
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        sentence_converted = tokenizer.encode(sentence, with_eos=True)
        tokens.append(sentence_converted)
        tokenlen.append(len(tokens[-1]))

    return np.array(tokens), np.array(tokenlen).astype(np.int32)

def main():
    # arguments
    args = parse_args()
    
    # define data generator
    train_libri_path = args.train_data_path
    dev_libri_path = args.dev_data_path
    train_texts, train_audio_path = data_preparation(train_libri_path)
    dev_texts, dev_audio_path = data_preparation(dev_libri_path)
    # save to corpus
    with open(args.corpus_path+"/train_gt.txt", 'w') as fout:
        fout.write("\n".join(train_texts))

    
    print("Process train/dev features...")
    if not os.path.exists(args.feat_path):
        os.makedirs(args.feat_path)

    # texts
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
    tokenizer = text_encoder(args.unit, special_tokens, args.corpus_path)
    train_chars, train_charlen = process_texts(
                                    special_tokens, 
                                    train_texts, 
                                    tokenizer)
    dev_chars, dev_charlen = process_texts(
                                    special_tokens, 
                                    dev_texts, 
                                    tokenizer)
    # save text features
    np.save(args.feat_path+"/train_{}s.npy".format(args.unit), train_chars)
    np.save(args.feat_path+"/train_{}len.npy".format(args.unit), train_charlen)
    np.save(args.feat_path+"/dev_{}s.npy".format(args.unit), dev_chars)
    np.save(args.feat_path+"/dev_{}len.npy".format(args.unit), dev_charlen)

    
    # audios
    print("Process train audios...")
    train_feats, train_featlen = process_audios(train_audio_path, args)
    np.save(args.feat_path+"/train_feats.npy", train_feats)    
    np.save(args.feat_path+"/train_featlen.npy", train_featlen)
    
    print("Process dev audios...")
    dev_feats, dev_featlen = process_audios(dev_audio_path, args)
    np.save(args.feat_path+"/dev_feats.npy", dev_feats)
    np.save(args.feat_path+"/dev_featlen.npy", dev_featlen)
    
    # augmentation
    if args.augmentation:   
        folder = args.feat_path.split("/")[1]
        speed_list = [1.1]
        volume_list = [0.4, 1.5]
        
        # speed aug
        for s in speed_list:
            aug_audio_path = speed_augmentation(filelist=train_audio_path,
                                                target_folder="data/{}/LibriSpeech_speed_aug".format(folder), 
                                                speed=s)
            aug_feats, aug_featlen = process_audios(aug_audio_path, args)
            np.save(args.feat_path+"/aug_feats_{}_{}.npy".format("speed",s), aug_feats)    
            np.save(args.feat_path+"/aug_featlen_{}_{}.npy".format("speed",s), aug_featlen)

        # volume aug
        aug_audio_path = volume_augmentation(filelist=train_audio_path,
                                            target_folder="data/{}/LibriSpeech_volume_aug".format(folder), 
                                            vol_range=volume_list)
        aug_feats, aug_featlen = process_audios(aug_audio_path, args)
        np.save(args.feat_path+"/aug_feats_{}.npy".format("vol"), aug_feats)    
        np.save(args.feat_path+"/aug_featlen_{}.npy".format("vol"), aug_featlen)
        
if __name__ == '__main__':
    main()
