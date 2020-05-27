import argparse

# bool type
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(
        description='A tensorflow implementation of end-to-end speech recognition system:'
                    'Listen, Attend and Spell (LAS)')    
    # feature arguments
    parser.add_argument('--dataset', 
                        type=str, 
                        default='LibriSpeech', 
                        help='Dataset: LibriSpeech or TEDLIUM.')
    parser.add_argument('--unit', 
                        type=str, 
                        default='subword', 
                        help='Encoding unit for texts processing.')
    parser.add_argument('--sample_rate', 
                        type=int, 
                        default=16000, 
                        help='Sample rate.')
    parser.add_argument('--feat_dim', 
                        type=int, 
                        default=39, 
                        help='The feature dimension.')
    parser.add_argument('--frame_length', 
                        type=int, 
                        default=25, 
                        help='Frame length in ms.')
    parser.add_argument('--frame_step', 
                        type=int, 
                        default=10, 
                        help='Frame step in ms.')
    parser.add_argument('--feat_type', 
                        type=str, 
                        default='mfcc', 
                        help='mfcc')
    parser.add_argument('--cmvn', 
                        type=str2bool, 
                        default=True, 
                        help='Apply cmvn or not.')
    parser.add_argument('--augmentation', 
                        type=str2bool, 
                        default=False, 
                        help='Apply data augmentation or not.')
    parser.add_argument('--split', 
                        type=str, 
                        default='dev', 
                        help='Split used for evaluation.')
    # training arguments
    parser.add_argument('--verbose', 
                        '-vb',
                        type=int, 
                        default=0, 
                        help='Verbosity.')
    parser.add_argument('--bucketing', 
                        type=str2bool, 
                        default=True, 
                        help='Apply bucketing.')
    parser.add_argument('--batch_size', 
                        '-bs',
                        type=int, 
                        default=32, 
                        help='The training batch size.')
    parser.add_argument('--lr', 
                        type=float, 
                        default=1e-3, 
                        help='The training learning rate.')
    parser.add_argument('--grad_clip', 
                        type=float, 
                        default=5, 
                        help='Apply gradient clipping.')
    parser.add_argument('--dropout_rate', 
                        type=float, 
                        default=0.5, 
                        help='The probability of drop out.')
    parser.add_argument('--epoch', 
                        type=int, 
                        default=10, 
                        help='The number of training epochs.')
    parser.add_argument('--restore_epoch', 
                        type=int, 
                        default=-1, 
                        help='The epoch you want to restore.')
    parser.add_argument('--label_smoothing', 
                        type=str2bool, 
                        default=True, 
                        help='Apply label smoothing.')
    parser.add_argument('--ctc', 
                        type=str2bool, 
                        default=False, 
                        help='Apply ctc.')
    parser.add_argument('--ctc_weight', 
                        type=float, 
                        default=0.2, 
                        help='Weighting of ctc.')
    # hparams of Listener
    parser.add_argument('--enc_units',
                        type=int,
                        default=64,
                        help='The hidden dimension of the pBLSTM in Listener.')
    parser.add_argument('--num_enc_layers',
                        type=int,
                        default=2,
                        help='The number of layers of pBLSTM in Listener.')
    # hparams of Speller
    parser.add_argument('--dec_units',
                        type=int,
                        default=128,
                        help='The hidden dimension of the LSTM in Speller.')
    parser.add_argument('--num_dec_layers',
                        type=int,
                        default=2,
                        help='The number of layers of LSTM in Speller.')
    parser.add_argument('--vocab_size',
                        type=int,
                        default=26,
                        help='Vocabulary size.')
    parser.add_argument('--embedding_size',
                        type=int,
                        default=128,
                        help='The dimension of the embedding matrix is: [vocab_size, embedding_size].')
    parser.add_argument('--attention_size',
                        type=int,
                        default=128,
                        help='Attention size.')
    parser.add_argument('--maxlen',
                        type=int,
                        default=400,
                        help='Max length of char sequences in training.')
    parser.add_argument('--warmup_step',
                        type=int,
                        default=500000,
                        help='Warmup step while training. During warmup step, teacher forcing rate is set to 1.')
    parser.add_argument('--max_step',
                        type=int,
                        default=500000,
                        help='Max step in scheduled sampling.')
    parser.add_argument('--min_rate',
                        type=float,
                        default=0.7,
                        help='Max step in scheduled sampling.')
    parser.add_argument('--teacher_forcing_rate',
                        type=float,
                        default=0.9,
                        help='Apply teacher forcing in decoder while training with constant sample rate.')
    # beam search
    parser.add_argument('--convert_rate',
                        type=float,
                        default=0.166,
                        help='Convert the length of audio to estimate the required decoding steps.')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='Size for beam search.')
    parser.add_argument('--apply_lm', 
                        type=str2bool, 
                        default=False, 
                        help='Apply language model.')
    parser.add_argument('--lm_weight', 
                        type=float, 
                        default=0.5, 
                        help='Weighting of recoring with language model.')
    # save dir
    parser.add_argument('--train_data_dir',
                        type=str,
                        default='./data/LibriSpeech/LibriSpeech_train/train-clean-100',
                        help='')
    parser.add_argument('--dev_data_dir',
                        type=str,
                        default='./data/LibriSpeech-100/LibriSpeech_dev/dev-clean',
                        help='')
    parser.add_argument('--test_data_dir',
                        type=str,
                        default='./data/LibriSpeech-100/LibriSpeech_test/test-clean',
                        help='')
    parser.add_argument('--feat_dir', 
                        type=str, 
                        default='./data/LibriSpeech/features', 
                        help='Path to save features.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./log',
                        help='Save log file..')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./model/las',
                        help='Save trained model.')
    parser.add_argument('--summary_dir',
                        type=str,
                        default='./summary',
                        help='Save summary.')

    args = parser.parse_args()

    return args
