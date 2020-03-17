import tensorflow as tf
import librosa
import speechpy
import numpy as np
from glob import glob
import string
import os
from tqdm import tqdm
from tokenizers import CharBPETokenizer
from las.arguments import parse_args
from utils.text import text_encoder
from utils.augmentation import speed_augmentation, volume_augmentation

_sample_threshold = 50000

def save_feats(threshold, cat, path, feats):
    """When number of feats > threshold, divide feature 
       into three parts to avoid memory error.
        
    Args:
        threshold: threshold to divide feats.
        path: save path.
        feats: extracted feature.
    """
    if len(feats) > threshold:
        n = len(feats) // 3 + 1
        # save
        for i in range(3):
            np.save(path+"/{}_feats_{}.npy".format(cat, i), feats[i*n:(i+1)*n])
    else:
        np.save(path+"/{}_feats.npy".format(cat), feats)

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

def process_ted(category, args, tokenizer):

    parent_path = 'data/TEDLIUM_release1/' + category + '/'
    train_text = []
    tokens, tokenlen, wave_files, offsets, durs = [], [], [], [], []

    # read STM file list
    stm = parent_path + 'stm/clean.stm'
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
            wave_file = parent_path + 'sph/%s.sph.wav' % field[0]
            wave_files.append(wave_file)

            # start, end info
            start, end = float(field[3]), float(field[4])
            offsets.append(start)
            durs.append(end - start)

    # save to corpus
    if category == "train" and args.unit == "subword":
        with open(args.corpus_path+"/train_gt.txt", 'w') as fout:
            fout.write("\n".join(train_texts))
        tokenizer.train_subword_tokenizer(args.corpus_path)

    feats = []
    featlen = []

    # setting                                                                                                                                                                                               
    frame_step = args.frame_step
    frame_length = args.frame_length
    feat_dim = args.feat_dim
    feat_type = args.feat_type
    cmvn = args.cmvn
     
    # save results
    for i, (wave_file, offset, dur) in enumerate(zip(wave_files, offsets, durs)):
        fn = "%s-%.2f" % (wave_file.split('/')[-1], offset)
        print("TEDLIUM corpus preprocessing (%d / %d) - '%s-%.2f]" % (i, len(wave_files), wave_file, offset))
        # load wave file
        if not os.path.exists(wave_file):
            sph_file = wave_file.rsplit('.',1)[0]
            if os.path.exists(sph_file):
                convert_sph(sph_file, wave_file)
            else:
                raise RuntimeError("Missing sph file from TedLium corpus at %s"%(sph_file))

        audio, fs = librosa.load(wave_file, mono=True, sr=None, offset=offset, duration=dur)
        #librosa.output.write_wav('test{}.wav'.format(i), audio, fs)
        
        if feat_type == "mfcc":
            # get mfcc feature
            mfcc = speechpy.feature.mfcc(audio, 
                                         fs, 
                                         frame_length=frame_length/1000, 
                                         frame_stride=frame_step/1000, 
                                         num_cepstral=feat_dim)
            if cmvn:
                mfcc = speechpy.processing.cmvn(mfcc, True)
            mfcc_39 = speechpy.feature.extract_derivative_feature(mfcc)
            mfcc_39 = mfcc_39.reshape(-1, 13*3).astype(np.float32)
            
            # save result ( exclude small mfcc data to prevent ctc loss )
            if len(tokens[i]) < mfcc_39.shape[0]:
                feats.append(mfcc_39)
                featlen.append(len(feats[-1]))
    
    return feats, featlen, tokens, tokenlen
        
def main_ted(args):

    print("Process train/dev features...")
    if not os.path.exists(args.feat_path):
        os.makedirs(args.feat_path)

    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']
    tokenizer = text_encoder(args.unit, special_tokens)

    for cat in ['train', 'dev', 'test']:
        feats, featlen, tokens, tokenlen = process_ted(cat, args, tokenizer)
        # save feats
        save_feats(_sample_threshold, cat, args.feat_path, feats)
        np.save(args.feat_path+"/{}_featlen.npy".format(cat), featlen)
        # save text features
        np.save(args.feat_path+"/{}_{}s.npy".format(cat, args.unit), tokens)
        np.save(args.feat_path+"/{}_{}len.npy".format(cat, args.unit), tokenlen)

def main_libri(args):

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
        # When number of feats > threshold, divide feature 
        # into three parts to avoid memory error.
        if len(audio_path) > _sample_threshold:
            featlen = []
            n = len(audio_path) // 3 + 1
            print("Process {} audios...".format(cat))
            for i in range(3):
                feats, featlen_ = process_audios(audio_path[i*n:(i+1)*n], args)
                featlen += featlen_
                # save
                np.save(args.feat_path+"/{}_feats_{}.npy".format(cat, i), feats)
        else:
            feats, featlen = process_audios(audio_path, args)
            np.save(args.feat_path+"/{}_feats.npy".format(cat ), feats)

        np.save(args.feat_path+"/{}_featlen.npy".format(cat), featlen)
        
        # augmentation
        if args.augmentation and cat == 'train':   
            folder = args.feat_path.split("/")[1]
            speed_list = [0.9, 1.1]
            volume_list = [0.8, 1.5]
            
            # speed aug
            for s in speed_list:
                aug_audio_path = speed_augmentation(filelist=audio_path,
                                                    target_folder="data/{}/LibriSpeech_speed_aug".format(folder), 
                                                    speed=s)
                # When number of feats > threshold, divide feature 
                # into three parts to avoid memory error.
                if len(aug_audio_path) > _sample_threshold:
                    aug_featlen = []
                    n = len(aug_audio_path) // 3 + 1
                    for i in range(3):
                        aug_feats, aug_featlen_ = process_audios(aug_audio_path[i*n:(i+1)*n], args)
                        aug_featlen += aug_featlen_
                        # save
                        np.save(args.feat_path+"/speed_{}_feats_{}.npy".format(s, i), feats)
                else:
                    aug_feats, aug_featlen = process_audios(aug_audio_path, args)
                    np.save(args.feat_path+"/speed_{}_feats.npy".format(s), feats)

                np.save(args.feat_path+"/{}_{}_featlen.npy".format("speed",s), aug_featlen)

            """
            ## Currently comment out vol augmentation:
            # volume aug
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

