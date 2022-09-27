# -*-coding:utf-8-*-

from ctypes import alignment
import sys

sys.path.append('../')
import torch
from torch.utils.data import DataLoader, Dataset
from utils import utils
from utils.slack import notification_slack
import numpy as np


class My_Dataloader():
    def __init__(self, vocab, random, seq_len=512,DataLoader=DataLoader):
        self.vocab = vocab
        self.DataLoader = DataLoader
        self.random = random
        self.visual_bbox = self._init_visual_bbox()
        self.seq_len = seq_len
        self.rng = random
    
    def __call__(self, dataset,  batch_size, shuffle):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, collate_fn=self._collate_fn)
    
    def _create_attention_mask(self, x):
        return torch.masked_fill(torch.ones(x.shape), x == self.vocab.index("<pad>"), 0)
    
    def _collate_fn(self, batch):
        output_dict = {}
        for i, b in enumerate(batch):
            #λ=1のポアソン分布からspanを生成
            batch[i]["mask_input_ids"], batch[i]["ml_position"], batch[i]["ml_label"] = utils.create_span_mask_for_ids(b["input_ids"], 0.3, 153, self.vocab, 1, self.rng)
            # if len(batch[i]["ml_position"]) == 0:
            #     notification_slack(f"maske lenght is 0!!!! and batch[i][input_ids]length is {len(batch[i]['input_ids'])}")   
        for i in ["mask_input_ids", "bbox", "pixel_values"]:
            padding_value=0
            if i == "mask_input_ids":
                padding_value = self.vocab.index("<pad>")
            output_dict[i] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(b[i]) for b in batch],
                batch_first=True,
                padding_value=padding_value
            )
            #pad_sequenceしても長さがseq_len以下の場合(not pixel values)
            if i != "pixel_values" and output_dict[i].shape[1] != self.seq_len:
                notification_slack(f"padding_{i}:{output_dict[i].shape} < 512, do pading")
                pad_len= self.seq_len -output_dict[i].shape[1]
                if i == "mask_input_ids":
                    #iput_ids > 0
                    pad_tensor = torch.ones((output_dict[i].shape[0], pad_len), dtype=torch.long)*padding_value
                else:
                    #bbox > [0, 0, 0, 0]
                    pad_tensor = torch.ones((output_dict[i].shape[0], pad_len, 4), dtype=torch.long)*padding_value

                output_dict[i] = torch.cat((output_dict[i], pad_tensor), dim=1)

        for i in ["ml_position", "ml_label"]:
            output_dict[i] = [torch.LongTensor(b[i]) for b in batch]

        output_dict["bool_mi_pos"] = torch.cat([b["bool_masked_pos"] for b in batch])
        output_dict["mi_label"] = [b["label"] for b in batch]

        attention_mask = self._create_attention_mask(output_dict["mask_input_ids"])
        output_dict["attention_mask"] = attention_mask
        
        #alignmentlabel for wpa
        al_labels = torch.nn.utils.rnn.pad_sequence(
            [b["alignment_labels"] for b in batch],
            batch_first=True,
            padding_value=False
        )
        if al_labels.shape[1] != self.seq_len:
            notification_slack(f"padding_alignment_labels:{al_labels.shape} < 512, do pading")
            pad_len= self.seq_len - al_labels.shape[1]
            pad_tensor = torch.zeros((al_labels.shape[0], pad_len)).to(torch.bool)
            al_labels = torch.cat((al_labels, pad_tensor), dim=1)
        
        output_dict["alignment_labels"] = al_labels

        #データセットにalignment_labelがない場合(ここで追加)
        # al_lable = self._crete_alignment_label(self.visual_bbox, output_dict["bbox"], output_dict["bool_mi_pos"])
        # output_dict["alignment_label"] = al_lable

        ##入力のため mask_input_ids => input_ids
        output_dict["input_ids"] = output_dict.pop("mask_input_ids")

        return output_dict

    def _init_visual_bbox(self, img_size=(14, 14), max_len=1000):
        #torch div : divide
        visual_bbox_x = torch.div(torch.arange(0, max_len * (img_size[1] + 1), max_len),
                                img_size[1], rounding_mode='trunc')
        visual_bbox_y = torch.div(torch.arange(0, max_len * (img_size[0] + 1), max_len),
                                img_size[0], rounding_mode='trunc')
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(img_size[0], 1),
                visual_bbox_y[:-1].repeat(img_size[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(img_size[0], 1),
                visual_bbox_y[1:].repeat(img_size[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)
        return visual_bbox
    
    #対応する画像がmaskされている 0 False, maskされていない: 1 True
    def _crete_alignment_label(self, visual_bbox, text_bboxes, bool_mi_pos):
        num_batch, num_text, _ = text_bboxes.shape
        if num_batch != bool_mi_pos.shape[0]:
            print("difarent batch size!")
        alignment_labels = []
        for i in range(num_batch):
            labels = torch.ones(num_text)
            for v_b in visual_bbox[bool_mi_pos[i]]:
                for j, t_b in enumerate(text_bboxes[i]):
                    if self._is_content_bbox(t_b, v_b) or self._is_content_bbox_2(t_b, v_b):
                        labels[j] = 0
            alignment_labels.append(labels.to(torch.bool))
        return torch.stack(alignment_labels)
    
    # (x0, y0, x1, y1) x0, y0比較
    def _is_content_bbox(self, text_bbox, image_bbox):
        if (text_bbox[0] >= image_bbox[0] and text_bbox[1] >= image_bbox[1] 
        and text_bbox[0] <= image_bbox[2] and text_bbox[1] <= image_bbox[3]):
            return True
        else:
            return False
        
    # (x0, y0, x1, y1) x1, y1比較
    def _is_content_bbox_2(self, text_bbox, image_bbox):
        if (text_bbox[2] >= image_bbox[0] and text_bbox[3] >= image_bbox[1] 
        and text_bbox[2] <= image_bbox[2] and text_bbox[3] <= image_bbox[3]):
            return True
        else:
            return False
    