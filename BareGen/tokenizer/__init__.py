from ..import_packages import *
from .ByteBPE import FastBBPE


class Tokenizer(object):
    def __init__(self, path=None):
        if path is None:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(curr_path, 'fastbpe_vocab.json')
        self.model = FastBBPE(vocab_size=2048, min_frequency=5, load=True, file_path=path)

    def __apply_std(self, token, dim):
        if len(token) >= dim - 2:
            tokens = token[:dim - 2]
            return [self.model.start_id] + tokens + [self.model.end_id]
        else:
            return [self.model.start_id] + token + [self.model.end_id] + [self.model.pad_id] * (dim - 2 - len(token))

    def __call__(self, x, std_out=False, dim=77, numpy=False):
        tokens = self.model.encode(x)
        if not std_out:
            return tokens
        std_out = None
        if isinstance(tokens[0], list):
            std_out= [self.__apply_std(token, dim) for token in tokens]
        else:
            std_out = self.__apply_std(tokens, dim)
        if numpy:
            return np.array(std_out)
        return std_out
    def decode(self, tokens):
        return self.model.decode(tokens)
