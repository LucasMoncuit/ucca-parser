import torch
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_sequence


class TensorDataSet(Data.Dataset):
    def __init__(self, *data):
        super(TensorDataSet, self).__init__()
        self.items = list(zip(*data))

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


def collate_fn(data):
    word_idx = list(zip(*data))[0]
    pos_idx = list(zip(*data))[1]
    dep_idx = list(zip(*data))[2]
    ent_idx = list(zip(*data))[3]
    ent_iob_idx = list(zip(*data))[4]
    passages = list(zip(*data))[5]
    trees = list(zip(*data))[6]
    all_nodes = list(zip(*data))[7]
    all_remote  = list(zip(*data))[8]
    alignments = list(zip(*data))[9]
    return (
        pad_sequence(word_idx, True),
        pad_sequence(pos_idx, True),
        pad_sequence(dep_idx, True),
        pad_sequence(ent_idx, True),
        pad_sequence(ent_iob_idx, True),
        passages,
        trees,
        all_nodes,
        all_remote,
        alignments,
    )