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
from tokenizers import CharBPETokenizer


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

def speed_augmentation(filelist, target_folder, speed):
    """Speed Augmentation
    
    Args:
        file_list: Path to audio files.
        target_folder: Folder of augmented audios.
        speed: Speed for augmentation.
    """
    audio_path = []
    aug_generator = sox.Transformer()
    print("Total audios:", len(filelist))
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

def volume_augmentation(filelist, target_folder, vol_range):
    """Volume Augmentation
    
    Args:
        file_list: Path to audio files.
        target_folder: Folder of augmented audios.
        volume_range: Range of volumes for augmentation.
    """
    audio_path = []
    aug_generator = sox.Transformer()
    print("Total audios:", len(filelist))
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for source_filename in tqdm(filelist):
        volume = np.around(
                np.random.uniform(vol_range[0],vol_range[1]), 2)
        aug_generator.vol(volume)
        file_id = source_filename.split("/")[-1]
        save_filename = target_folder+"/"+file_id.split(".")[0]+"_"+str(volume)+"."+file_id.split(".")[1] 
        aug_generator.build(source_filename, save_filename)
        audio_path.append(save_filename)
    return audio_path

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

def lookup_dicts(special_tokens):
    """
    Args:
        special_tokens: special charactors, <PAD>, <SOS>, <EOS>, <SPACE>
    Returns:
        char2id: dict, from character to index.
        id2char: dict, from index to character.
    """

    alphas = list(string.ascii_uppercase[:26])
    chars = special_tokens + alphas
    char2id = {}
    id2char = {}
    for i, c in enumerate(chars):
        char2id[c] = i
        id2char[i] = c
    return char2id, id2char

class text_encoder:
    "char tokenizarion and subword tokenization" 
    def __init__(self, unit, special_tokens, corpus_path):
        """
        Args:
            unit: unit used for encoding strings, char or subword.
            special_tokens: special charactors, <PAD>, <SOS>, <EOS>, <SPACE>
            corpus_path: path of training corpus.
        """
        self.unit = unit
        self.special_tokens = special_tokens

        # define char2id and id2char used in _encode_chars()
        if self.unit == "char":
            self.char2id, self.id2char = lookup_dicts(special_tokens)
            self.encode = self._encode_chars
            self.id_to_token = self._id_to_char()

        # utilize "tokenizers" library
        elif self.unit == "subword":
            self.train_subword_tokenizer(corpus_path)
            self.encode = self._encode_subwords
            self.id_to_token = self._id_to_subword()
        else:
            raise Exception('Unit not support!') 

    def get_vocab_size(self):
        if self.unit == "char":
            return len(self.id2char)
        else:
            return self.subword_tokenizer.get_vocab_size()   
        
    def _encode_chars(self, sentence, with_eos):
        """ 
        Args:
            sentence: str, texts to be encoded.
            with_eos: end with <EOS> token.
        Returns:
            tokens: list, encoded sequence.
        """
        tokens = [self.char2id[char] if char != ' ' else self.char2id['<SPACE>'] for char in list(sentence)]
        if with_eos:
            tokens += [self.char2id['<EOS>']]
        return tokens

    def _encode_subwords(self, sentence, with_eos):
        """ 
        Args:
            sentence: str, texts to be encoded.
            with_eos: end with <EOS> token.
        Returns:
            tokens: list, encoded sequence.
        """
        tokens = self.subword_tokenizer.encode(sentence).ids
        if with_eos:
            tokens += self.subword_tokenizer.encode("<EOS>").ids
        return tokens

    def _id_to_char(self):
        return self.id2char

    def _id_to_subword(self):
        id2subword = {}
        for i in range(self.get_vocab_size()):
            id2subword[self.subword_tokenizer.id_to_token(i)] = i
        return id2subword

    def train_subword_tokenizer(self, path):
        """Train subword tokenizers for subword encoding
        ref: https://github.com/huggingface/tokenizers
        """
        try:
            tokenizer = CharBPETokenizer(vocab_file=path+"bpe-vocab.json", merges_file=path+"bpe-merges.txt")
        except:
            tokenizer = CharBPETokenizer()
            tokenizer.train(
                [path+"/train_gt.txt"],
                vocab_size=500,
                min_frequency=2,
                show_progress=True,
                special_tokens=self.special_tokens[:3]+["<unk>"],
            )
            tokenizer.save(path, "bpe")
        self.subword_tokenizer = tokenizer

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
    

    # augmentation
    if args.augmentation:   
        speed_list = [0.9, 1.1]
        volume_list = [0.2, 2]
        
        # speed aug
        for s in speed_list:
            aug_audio_path = speed_augmentation(filelist=train_audio_path,
                                                target_folder="data/LibriSpeech/LibriSpeech_speed_aug", 
                                                speed=s)
            aug_feats, aug_featlen = process_audios(aug_audio_path,
                                                        frame_step=10, 
                                                        frame_length=25,
                                                        feat_dim=13,
                                                        feat_type='mfcc',
                                                        cmvn=True)
            np.save(args.feat_path+"/aug_feats_{}_{}.npy".format("speed",s), aug_feats)    
            np.save(args.feat_path+"/aug_featlen_{}_{}.npy".format("speed",s), aug_featlen)
        
        # volume aug
        aug_audio_path = volume_augmentation(filelist=train_audio_path,
                                            target_folder="data/LibriSpeech/LibriSpeech_volume_aug", 
                                            vol_range=volume_list)
        aug_feats, aug_featlen = process_audios(aug_audio_path,
                                                    frame_step=10, 
                                                    frame_length=25,
                                                    feat_dim=13,
                                                    feat_type='mfcc',
                                                    cmvn=True)
        np.save(args.feat_path+"/aug_feats_{}.npy".format("vol"), aug_feats)    
        np.save(args.feat_path+"/aug_featlen_{}.npy".format("vol"), aug_featlen)

if __name__ == '__main__':
    main()
