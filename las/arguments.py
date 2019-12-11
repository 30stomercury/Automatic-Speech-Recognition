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
    parser.add_argument('--sample_rate', 
                        type=int, 
                        default=22050, 
                        help='Sample rate.')
    parser.add_argument('--feat_dim', 
                        type=int, 
                        default=40, 
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
                        default='fbank', 
                        help='Log-mel filter bank.')
    parser.add_argument('--feat_path', 
                        type=str, 
                        default='./data/features', 
                        help='Path to save features.')
    # training arguments
    parser.add_argument('--is_training', 
                        type=str2bool, 
                        default=True, 
                        help='Whether model is in training phase.')
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
    parser.add_argument('--keep_proba', 
                        type=float, 
                        default=0.5, 
                        help='The keep probability of drop out.')
    parser.add_argument('--epoch', 
                        type=int, 
                        default=10, 
                        help='The number of training epochs.')
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
    parser.add_argument('--maxlen',
                        type=int,
                        default=400,
                        help='Max length of char sequences in training.')
    parser.add_argument('--convert_rate',
                        type=float,
                        default=0.166,
                        help='Convert the length of audio estimate the length of chars.')
    parser.add_argument('--teacher_forcing_rate',
                        type=float,
                        default=0.1,
                        help='Apply teacher forcing in decoder while training with constant sample rate.')
    # save path
    parser.add_argument('--result_path',
                        type=str,
                        default='./results',
                        help='Save predicted texts.')
    parser.add_argument('--save_path',
                        type=str,
                        default='./model/las',
                        help='Save trained model.')
    parser.add_argument('--summary_path',
                        type=str,
                        default='./summary',
                        help='Save summary.')

    args = parser.parse_args()

    return args
