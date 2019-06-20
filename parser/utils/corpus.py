import os

import torch

from ucca import core, layer0, textutil
from ucca.convert import xml2passage
from ucca.textutil import annotate_all
from .dataset import TensorDataSet
from .instance import Instance
from tqdm import tqdm

def from_text(text, passage_id="1", tokenized=False, one_per_line=False, extra_format=None, lang="en", *args, **kwargs):
    """Converts from tokenized strings to a Passage object.

    :param text: a multi-line string or a sequence of strings:
                 each line will be a new paragraph, and blank lines separate passages
    :param passage_id: prefix of ID to set for returned passages
    :param tokenized: whether the text is already given as a list of tokens
    :param one_per_line: each line will be a new passage rather than just a new paragraph
    :param extra_format: value to set in passage.extra["format"]
    :param lang: language to use for tokenization model

    :return: generator of Passage object with only Terminal units
    """
    del args, kwargs
    if isinstance(text, str):
        text = text.splitlines()
    if tokenized:
        text = (text,)  # text is a list of tokens, not list of lines
    p = l0 = paragraph = None
    i = 0
    for line in text:
        if not tokenized:
            line = line.strip()
        if line or one_per_line:
            if p is None:
                p = core.Passage("%s_%d" % (passage_id, i), attrib=dict(lang=lang))
                if extra_format is not None:
                    p.extra["format"] = extra_format
                l0 = layer0.Layer0(p)
                #layer1.Layer1(p)
                paragraph = 1
            for lex in textutil.get_tokenizer(tokenized, lang=lang)(line):
                l0.add_terminal(text=lex.orth_, punct=lex.is_punct, paragraph=paragraph)
            paragraph += 1
        if p and (not line or one_per_line):
            yield p
            p = None
            i += 1
    if p:
        yield p

class Corpus(object):
    def __init__(self, dic_name=None, lang=None):
        self.dic_name = dic_name
        self.language = lang
        self.passages = self.read_passages(dic_name)
        self.instances = [Instance(passage) for passage in self.passages]

    @property
    def num_sentences(self):
        return len(self.passages)
    
    @property
    def lang(self):
        return self.language

    def __repr__(self):
        return "%s : %d sentences, %s language" % (self.dic_name, self.num_sentences, self.lang)

    def __getitem(self, index):
        return self.passages[index]

    def read_passages(self, path):
        passages = []
        if os.path.isdir(path):
            for file in sorted(os.listdir(path)):
                file_path = os.path.join(path, file)
                if os.path.isdir(file_path):
                    print(file_path)
                passages.append(xml2passage(file_path))
        else:  # text file, not a directory
            with open(path, encoding="utf-8") as f:
                for passage in annotate_all(from_text(tqdm(list(f)), lang=self.lang, one_per_line=True), lang=self.lang):
                    passages.append(passage)
        return passages

    def generate_inputs(self, vocab, is_training=False):
        lang_idxs, word_idxs = [], []
        pos_idxs, dep_idxs, ent_idxs, ent_iob_idxs = [], [], [], []
        trees, all_nodes, all_remote = [], [], []
        for instance in self.instances:
            _word_idxs = vocab.word2id([vocab.START] + instance.words + [vocab.STOP])
            _pos_idxs = vocab.pos2id([vocab.START] + instance.pos + [vocab.STOP])
            _dep_idxs = vocab.dep2id([vocab.START] + instance.dep + [vocab.STOP])
            _entity_idxs = vocab.entity2id([vocab.START] + instance.ent + [vocab.STOP])
            _iob_idxs = vocab.ent_iob2id([vocab.START] + instance.ent_iob + [vocab.STOP])
            _lang_idxs = [vocab.lang2id(self.lang)] * len(_word_idxs)

            nodes, (heads, deps, labels) = instance.gerenate_remote()
            if len(heads) == 0:
                _remotes = ()
            else:
                heads, deps = torch.tensor(heads), torch.tensor(deps)
                labels = [[vocab.edge_label2id(l) for l in label] for label in labels]
                labels = torch.tensor(labels)
                _remotes = (heads, deps, labels)

            lang_idxs.append(torch.tensor(_lang_idxs))
            word_idxs.append(torch.tensor(_word_idxs))
            ent_iob_idxs.append(torch.tensor(_iob_idxs))
            pos_idxs.append(torch.tensor(_pos_idxs))
            dep_idxs.append(torch.tensor(_dep_idxs))
            ent_idxs.append(torch.tensor(_entity_idxs))

            if is_training:
                trees.append(instance.tree)
                all_nodes.append(nodes)
                all_remote.append(_remotes)
            else:
                trees.append([])
                all_nodes.append([])
                all_remote.append([])

        return TensorDataSet(
            lang_idxs,
            word_idxs,
            pos_idxs,
            dep_idxs,
            ent_idxs,
            ent_iob_idxs,
            self.passages,
            trees,
            all_nodes,
            all_remote,
        )


class Embedding(object):
    def __init__(self, words, vectors):
        super(Embedding, self).__init__()

        self.words = words
        self.vectors = vectors
        self.pretrained = {w: v for w, v in zip(words, vectors)}

    def __len__(self):
        return len(self.words)

    def __contains__(self, word):
        return word in self.pretrained

    def __getitem__(self, word):
        return self.pretrained[word]

    @property
    def dim(self):
        return len(self.vectors[0])

    @classmethod
    def load(cls, fname, smooth=True):
        with open(fname, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines[1:]]
        reprs = [(s[0], list(map(float, s[1:]))) for s in splits]
        words, vectors = map(list, zip(*reprs))
        vectors = torch.tensor(vectors)
        if smooth:
            vectors /= torch.std(vectors)
        embedding = cls(words, vectors)

        return embedding