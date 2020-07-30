import string
from tokenizers import CharBPETokenizer

SPECIAL_TOKENS = ['<PAD>', '<SOS>', '<EOS>', '<SPACE>']

def lookup_dicts(special_tokens):
    """
    Args:
        special_tokens: special charactors, <PAD>, <SOS>, <EOS>, <SPACE>
    Returns:
        char2id: dict, from character to index.
        id2char: dict, from index to character.
    """

    alphas = list(string.ascii_uppercase[:26])
    tokens = special_tokens + alphas
    token_to_id = {}
    id_to_token = {}
    for i, c in enumerate(tokens):
        token_to_id[c] = i
        id_to_token[i] = c

    return token_to_id, id_to_token


def train_subword_tokenizer(size, special_tokens, path):
    """Train subword tokenizers for subword encoding
    ref: https://github.com/huggingface/tokenizers

    Args:
        path: path of training corpus.
    """
    tokenizer = CharBPETokenizer()
    tokenizer.train(
        [path+"/corpus_all.txt"],
        vocab_size=size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens[:3]+["<unk>"],
    )
    tokenizer.save(path, "bpe")

class Subword_Encoder:
    "Subword tokenization" 

    def __init__(self, path='subword/'):
        """ 
        Args:
            path: str, a path to vocab file.
        """
        
        # Load vocab
        self.subword_tokenizer = CharBPETokenizer(vocab_file=path+"/bpe-vocab.json", merges_file=path+"/bpe-merges.txt")

        self.encode = self._encode_subwords
        self.id_to_token = self._id_to_subword()
        self.token_to_id = self._subword_to_id()

    def get_vocab_size(self):
        return self.subword_tokenizer.get_vocab_size()   

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
            tokens += [2] # 2 is the id of <EOS> token
        return tokens

    def _id_to_subword(self):
        id2subword = {}
        for i in range(self.get_vocab_size()):
            id2subword[i] = self.subword_tokenizer.id_to_token(i)
        return id2subword

    def _subword_to_id(self):
        subword2id = {}
        for i in range(self.get_vocab_size()):
            subword2id[self.subword_tokenizer.id_to_token(i)] = i
        return subword2id

class Char_Encoder:
    "Char tokenizarion"

    def __init__(self):
        """
        Args:
            special_tokens: special charactors, <PAD>, <SOS>, <EOS>, <SPACE>
        """

        # define char2id and id2char used in _encode_chars()
        self.char2id, self.id2char = lookup_dicts(SPECIAL_TOKENS)
        self.encode = self._encode_chars
        self.id_to_token = self.id2char
        self.token_to_id = self.char2id


    def get_vocab_size(self):
        return len(self.id2char)
        
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
