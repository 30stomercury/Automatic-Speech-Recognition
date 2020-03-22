import tensorflow as tf
import librosa
import soundfile
import speechpy
import joblib
import numpy as np
from glob import glob
import string
import os
from tqdm import tqdm
from las.arguments import parse_args
from utils.text import text_encoder
from utils.augmentation import speed_augmentation, volume_augmentation

# When number of audios in a set (usually training set) > threshold, divide set into several parts to avoid memory error.
_SAMPLE_THRESHOLD = 50000

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
            audio, fs = soundfile.read(p)
        except:
            print(p)
            pass

        if feat_type == 'mfcc':
            assert feat_dim == 13, "13 is commonly used"
            mfcc = speechpy.feature.mfcc(audio, 
                                         fs, 
                                         frame_length=frame_length/1000, 
                                         frame_stride=frame_step/1000, 
                                         num_cepstral=feat_dim)
            
            if cmvn:
                mfcc = speechpy.processing.cmvn(mfcc, True)
            mfcc_39 = speechpy.feature.extract_derivative_feature(mfcc)
            feats.append(mfcc_39.reshape(-1, feat_dim*3).astype(np.float32))

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

    return feats, featlen

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

def process_ted(cat, args, tokenizer):
    """Process ted audios"""

    _parent_path = 'data/TEDLIUM_release1/' + cat + '/'
    train_text = []
    tokens, tokenlen, wave_files, offsets, durs = [], [], [], [], []

    # read STM file list
    stm = _parent_path + 'stm/clean.stm'
    with open(stm, 'rt') as f:
        records = f.readlines()
        for record in records:
            field = record.split()

            # label index
            sentence = ' '.join(field[6:-1]).upper()
            sentence = sentence.translate(
                            str.maketrans('', '', string.punctuation+'1234567890'))
            sentence_converted = tokenizer.encode(sentence, with_eos=True)

            if len(sentence_converted) < 2:
                continue            

            tokens.append(sentence_converted)
            tokenlen.append(len(tokens[-1]))
            train_text.append(sentence)

            # wave file name
            wave_file = _parent_path + 'sph/%s.sph.wav' % field[0]
            wave_files.append(wave_file)

            # start, end info
            start, end = float(field[3]), float(field[4])
            offsets.append(start)
            durs.append(end - start)

    # save to corpus
    if cat == "train" and args.unit == "subword":
        with open(args.corpus_path+"/train_gt.txt", 'w') as fout:
            fout.write("\n".join(train_texts))
        tokenizer.train_subword_tokenizer(args.corpus_path)

    audio_path = []
     
    # save results
    for i, (wave_file, offset, dur) in enumerate(zip(wave_files, offsets, durs)):
        fn = "%s-%.2f" % (wave_file.split('/')[-1], offset)
        print("TEDLIUM corpus preprocessing (%d / %d) - '%s-%.2f]" % (i, len(wave_files), wave_file, offset))

        # load
        audio, fs = librosa.load(wave_file, mono=True, sr=None, offset=offset, duration=dur)

        # for debug and augmentation
        train_wav_path = 'data/TEDLIUM_release1/{}_wav'.format(cat)
        file_name = wave_file.split("/")[-1][:-8].replace(".", "")
        file_path = train_wav_path+'/{}_{}.wav'.format(file_name, i)

        if not os.path.exists(train_wav_path):
            os.makedirs(train_wav_path)
        librosa.output.write_wav(file_path, audio, fs)

        audio_path.append(file_path)
        
    # process audio        
    feats, featlen = process_audios(audio_path, args)

    return feats, featlen, tokens, tokenlen, audio_path
        
def main_ted(args):

    def save_ted_feats(feats, cat, k):
        """When number of feats > threshold, divide feature 
           into three parts to avoid memory error.
            
        Args:
            feats: extracted features.
        """
        if len(feats) > _SAMPLE_THRESHOLD:
            n = len(feats) // k + 1
            # save
            for i in range(k):
                joblib.dump(feats[i*n:(i+1)*n], args.feat_path+"/{}_feats_{}.pkl".format(cat, i))
        else:
            joblib.dump(feats, args.feat_path+"/{}_feats.pkl".format(cat))

    print("Process train/dev features...")
    if not os.path.exists(args.feat_path):
        os.makedirs(args.feat_path)

    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
    tokenizer = text_encoder(args.unit, special_tokens)

    for cat in ['train', 'dev', 'test']:
        
        print("Process {} data...".format(cat))
        feats, featlen, tokens, tokenlen, audio_path = process_ted(cat, args, tokenizer)

        # save feats
        save_ted_feats(feats, cat, 3)
        np.save(args.feat_path+"/{}_featlen.npy".format(cat), featlen)

        # save text features
        np.save(args.feat_path+"/{}_{}s.npy".format(cat, args.unit), tokens)
        np.save(args.feat_path+"/{}_{}len.npy".format(cat, args.unit), tokenlen)
        
        feats = []

        # augmentation
        if args.augmentation and cat == "train":

            speed_list = [0.9, 1.1]
            print("Process aug data...") 
            # speed aug
            for s in speed_list:
                aug_audio_path = speed_augmentation(filelist=audio_path,
                                                    target_folder="data/TEDLIUM_release1/ted_speed_aug", 
                                                    speed=s)
                aug_feats, aug_featlen = process_audios(aug_audio_path, args) 
                save_ted_feats(aug_feats, "speed_{}".format(s), 3)
                np.save(args.feat_path+"/speed_{}_featlen.npy".format(s), aug_featlen)
                aug_feats = []

def main_libri(args):

    def process_libri_feats(audio_path, cat, k):
        """When number of feats > threshold, divide feature 
           into several parts to avoid memory error.
        """
        if len(audio_path) > _SAMPLE_THRESHOLD:
            featlen = []
            n = len(audio_path) // k + 1
            print("Process {} audios...".format(cat))
            for i in range(k):
                #feats, featlen_ = process_audios(audio_path[i*n:(i+1)*n], args)
                feats, featlen_ = process_audios(audio_path, args)
                featlen += featlen_
                # save
                joblib.dump(feats, args.feat_path+"/{}_feats_{}.pkl".format(cat, i))
                feats = []
        else:
            feats, featlen = process_audios(audio_path, args)
            joblib.dump(feats, args.feat_path+"/{}_feats.pkl".format(cat))

        np.save(args.feat_path+"/{}_featlen.npy".format(cat), featlen)


    # texts
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
    tokenizer = text_encoder(args.unit, special_tokens)
    
    path = [args.train_data_path, args.dev_data_path, args.test_data_path]
    for index, cat in enumerate(['train', 'dev', 'test']):
        # prepare data
        libri_path = path[index]
        texts, audio_path = data_preparation(libri_path)

        if cat == 'train' and args.unit == 'subword':
            # save to corpus
            with open(args.corpus_path+"/train_gt.txt", 'w') as fout:
                fout.write("\n".join(texts))
            # train BPE
            tokenizer.train_subword_tokenizer(args.corpus_path)

        print("Process {} texts...".format(cat))
        if not os.path.exists(args.feat_path):
            os.makedirs(args.feat_path)

        tokens, tokenlen = process_texts(texts, tokenizer)

        # save text features
        np.save(args.feat_path+"/{}_{}s.npy".format(cat, args.unit), tokens)
        np.save(args.feat_path+"/{}_{}len.npy".format(cat, args.unit), tokenlen)

        # audios
        process_libri_feats(audio_path, cat, 4)
        
        # augmentation
        if args.augmentation and cat == 'train':   
            folder = args.feat_path.split("/")[1]
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
            np.save(args.feat_path+"/aug_feats_{}.npy".format("vol"), aug_feats)    
            np.save(args.feat_path+"/aug_featlen_{}.npy".format("vol"), aug_featlen)
            """

if __name__ == '__main__':
    # arguments
    args = parse_args()

    if args.dataset == 'LibriSpeech':
        main_libri(args)

    elif args.dataset == 'TEDLIUM':
        main_ted(args)

    else:
        print("Set dataset to 'Librispeech' or 'TEDLIUM'")
