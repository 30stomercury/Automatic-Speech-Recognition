import joblib
from glob import glob
import string
import os
import sys
import json
import logging
from tqdm import tqdm
import soundfile as sf
import speechpy
import numpy as np
from las.arguments import parse_args
from utils.tokenizer import SubwordEncoder, CharEncoder
from utils.augmentation import SpeedAugmentation, VolumeAugmentation

# When number of audios in a set (usually training set) > threshold, divide set into several parts to avoid memory error.
_SAMPLE_THRESHOLD = 50000

# set logging
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)s:%(message)s', 
                    level=logging.INFO,
                    datefmt='%I:%M:%S')


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
        
        audio, fs = sf.read(p)

        if feat_type == 'mfcc':
            assert feat_dim == 39, "13+delta+accelerate"
            mfcc = speechpy.feature.mfcc(audio, 
                                         fs, 
                                         frame_length=frame_length/1000, 
                                         frame_stride=frame_step/1000, 
                                         num_cepstral=13) # 13 is commonly used
            
            if cmvn:
                mfcc = speechpy.processing.cmvn(mfcc, True)
            mfcc_39 = speechpy.feature.extract_derivative_feature(mfcc)
            feats.append(mfcc_39.reshape(-1, feat_dim).astype(np.float32))

        elif feat_type == 'fbank':
            fbank = speechpy.feature.lmfe(audio, 
                                          fs, 
                                          frame_length=frame_length/1000, 
                                          frame_stride=frame_step/1000, 
                                          num_filters=feat_dim)
            if cmvn:
                fbank = speechpy.processing.cmvn(fbank, True)
            feats.append(fbank.reshape(-1, feat_dim).astype(np.float32))

        featlen.append(len(feats[-1]))

    return np.array(feats), featlen

def process_texts(texts, tokenizer):
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
        

def main_libri(args, tokenizer):

    def process_libri_feats(audio_path, cat, k):
        """When number of feats > threshold, divide feature 
           into several parts to avoid memory error.
        """
        if len(audio_path) > _SAMPLE_THRESHOLD:
            featlen = []
            n = len(audio_path) // k + 1
            logging.info("Process {} audios...".format(cat))
            for i in range(k):
                feats, featlen_ = process_audios(audio_path[i*n:(i+1)*n], args)
                featlen += featlen_
                # save
                joblib.dump(feats, args.feat_dir+"/{}-feats-{}.pkl".format(cat, i))
                feats = []
        else:
            feats, featlen = process_audios(audio_path, args)
            joblib.dump(feats, args.feat_dir+"/{}-feats.pkl".format(cat))

        np.save(args.feat_dir+"/{}-featlen.npy".format(cat), featlen)

    # data directories
    path = [('train-100', args.train_100hr_corpus_dir), ('train-360', args.train_360hr_corpus_dir), 
            ('train-500', args.train_500hr_corpus_dir),
            ('dev',args.dev_data_dir), ('test', args.test_data_dir)]
    path = [('dev',args.dev_data_dir), ('test', args.test_data_dir)]

    for element in path:
                     
        # prepare data
        cat = element[0]            # the prefix of filenames
        libri_path = element[1]
        texts, audio_path = data_preparation(libri_path)

        logging.info("Process {} texts...".format(libri_path))
        if not os.path.exists(args.feat_dir):
            os.makedirs(args.feat_dir)
        
        tokens, tokenlen = process_texts(texts, tokenizer)

        # save text features
        np.save(args.feat_dir+"/{}-{}s.npy".format(cat, args.unit), tokens)
        np.save(args.feat_dir+"/{}-{}len.npy".format(cat, args.unit), tokenlen)
        
        # audios
        #process_libri_feats(audio_path, cat, 4)
        
        # augmentation
        if args.augmentation and 'train' in cat:   
            folder = args.feat_dir.split("/")[1]
            speed_list = [0.9, 1.1]
            
            # speed aug
            for s in speed_list:
                aug_audio_path = speed_augmentation(filelist=audio_path,
                                                    target_folder="data/{}/LibriSpeech_speed_aug".format(folder), 
                                                    speed=s)
                process_libri_feats(aug_audio_path, "speed_{}".format(s), 4)

            """Currently comment out vol augmentation:
            # volume aug
            volume_list = [0.8, 1.5]
            aug_audio_path = volume_augmentation(filelist=train_audio_path,
                                                target_folder="data/{}/LibriSpeech_volume_aug".format(folder), 
                                                vol_range=volume_list)
            aug_feats, aug_featlen = process_audios(aug_audio_path, args)
            # save
            np.save(args.feat_dir+"/aug_feats_{}.npy".format("vol"), aug_feats)    
            np.save(args.feat_dir+"/aug_featlen_{}.npy".format("vol"), aug_featlen)
            """

if __name__ == '__main__':

    # arguments
    args = parse_args()

    print('=' * 60 + '\n')
    logging.info('Parameters are:\n%s\n', json.dumps(vars(args), sort_keys=False, indent=4))
    print('=' * 60 )

    # Choose unit

    if args.unit == 'char':
        logging.info('Using {} tokenizer.'.format(args.unit))
        tokenizer = CharEncoder()

    elif args.unit == 'subword':
        logging.info('Using {} tokenizer: {}'.format(args.unit, args.subword_dir))
        tokenizer = SubwordEncoder(args.subword_dir)


    assert args.dataset == 'LibriSpeech'
    main_libri(args, tokenizer)

