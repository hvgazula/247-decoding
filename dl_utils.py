import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Brain2TextDataset(Dataset):
    """Brainwave-to-English Dataset.
       Pytorch Dataset wrapper
    """
    def __init__(self, signals, labels):
        """
        Args:
            signals (list): brainwave examples.
            labels (list): english examples.
        """
        assert (len(signals) == len(labels))
        example_list = sorted(list(zip(signals, labels)), key=self.getKey)
        example_list = [(torch.from_numpy(k[0]).float(),
                         torch.tensor(k[1]).long()) for k in example_list]
        self.examples = example_list

    def getKey(self, item):
        return len(item[0])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class MyCollator(object):
    def __init__(self, CONFIG, vocabulary):
        self.CONFIG = CONFIG
        self.vocabulary = vocabulary
        self.pad_token = CONFIG["pad_token"]

    def __call__(self, batch):
        # do something with batch and self.params
        src = pad_sequence([batch[i][0] for i in range(len(batch))],
                           batch_first=True,
                           padding_value=0.)
        labels = pad_sequence([batch[i][1] for i in range(len(batch))],
                              batch_first=True,
                              padding_value=self.vocabulary[self.pad_token])
        trg = torch.zeros(labels.size(0), labels.size(1),
                          len(self.vocabulary)).scatter_(
                              2, labels.unsqueeze(-1), 1)
        trg, trg_y = trg[:, :-1, :], labels[:, 1:]
        pos_mask, pad_mask = self.masks(trg_y)
        return src, trg, trg_y, pos_mask, pad_mask

    def masks(self, labels):
        pos_mask = (torch.triu(torch.ones(labels.size(1),
                                          labels.size(1))) == 1).transpose(
                                              0, 1).unsqueeze(0)
        pos_mask = pos_mask.float().masked_fill(pos_mask == 0,
                                                float('-inf')).masked_fill(
                                                    pos_mask == 1, float(0.0))
        pad_mask = labels == self.vocabulary[self.pad_token]
        return pos_mask, pad_mask
