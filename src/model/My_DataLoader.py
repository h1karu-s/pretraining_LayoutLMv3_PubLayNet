# -*-coding:utf-8-*-

import sys
import random

sys.path.append('../')
import torch
from torch.utils.data import DataLoader, Dataset
from utils import utils

random.seed(314)

class My_Dataloader():
    def __init__(self, vocab, DataLoader=DataLoader):
        self.vocab = vocab
        self.DataLoader = DataLoader
        self.random = random
    
    def __call__(self, dataset,  batch_size, shuffle):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self._collate_fn)
    
    def _create_attention_mask(self, x):
        return torch.masked_fill(torch.ones(x.shape), x == self.vocab.index("<pad>"), 0)
    
    def _collate_fn(self, batch):
        output_dict = {}
        for i, b in enumerate(batch):
            batch[i]["input_ids"], batch[i]["mask_position"], batch[i]["mask_label"] = utils.create_span_mask_for_ids(b["input_ids"], 0.3, 153, self.vocab, 3, self.random)
        for i in ["input_ids", "bbox", "pixel_values"]:
            padding_value=0
            if i == "input_ids":
                padding_value = self.vocab.index("<pad>")
            output_dict[i] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(b[i]) for b in batch],
                batch_first=True,
                padding_value=padding_value
            )
        for i in ["mask_position", "mask_label"]:
            output_dict[i] = [torch.LongTensor(b[i]) for b in batch]

        attention_mask = self._create_attention_mask(output_dict["input_ids"])
        output_dict["attention_mask"] = attention_mask
        return output_dict
