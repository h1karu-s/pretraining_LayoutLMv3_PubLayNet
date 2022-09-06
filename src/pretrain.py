# -*-coding:utf-8-*-

import sys
import argparse
import os
import random
import  pickle
import math

import torch
from transformers import AutoConfig, RobertaModel, LayoutLMv3Tokenizer
from model import LayoutLMv3forMLM, My_DataLoader
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup


def main(args):
    if not torch.cuda.is_available():
        raise ValueError("GPU is not available.")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device_ids = list(range(torch.cuda.device_count()))

    tokenizer = LayoutLMv3Tokenizer(f"{args.tokenizer_vocab_dir}vocab.json", f"{args.tokenizer_vocab_dir}merges.txt")
    ids = range(tokenizer.vocab_size)
    vocab = tokenizer.convert_ids_to_tokens(ids)

    if not args.model_params is None:
        model = torch.load(args.model_params)
    else:
        config = AutoConfig.from_pretrained(args.model_name)
        model = LayoutLMv3forMLM.LayoutLMv3ForMLM(config)
        Roberta_model = RobertaModel.from_pretrained("roberta-base")
        ## embedidng 層の重みをRobertaの重みで初期化
        weight_size = model.state_dict()["model.embeddings.word_embeddings.weight"].shape
        for i in range(weight_size[0]):
          model.state_dict()["model.embeddings.word_embeddings.weight"][i] = \
          Roberta_model.state_dict()["embeddings.word_embeddings.weight"][i]
    #optimizer 
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2, betas=(0.9, 0.98))
    #cross entropy
    loss_fn = torch.nn.CrossEntropyLoss()
    #modelをGPUへ
    model = torch.nn.DataParallel(model, device_ids = device_ids)
    model = model.to(f'cuda:{model.device_ids[0]}')
    #load input_file
    with open(args.input_file, 'rb') as f:
        data = pickle.load(f)
    #divide into train and valid
    n_train = math.floor(len(data) * args.ratio_train)
    train_data = data[:n_train]
    valid_data = data[n_train:]
    #create dataloader
    my_dataloader = My_DataLoader.My_Dataloader(vocab)
    train_dataloader = my_dataloader(train_data, batch_size=args.batch_size, shuffle=False)
    valid_dataloader = my_dataloader(valid_data, batch_size=args.batch_size, shuffle=False)

 

    #scheduler warm up lineary over fist 0.4% step
    iter_per_epoch = len(train_dataloader)
    num_warmup_steps = round((iter_per_epoch * args.max_epochs) * 0.048)
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
    
    #define calculation loss funciton
    def cal_loss(logits, batch):
            t = []
            for i in range(len(batch["mask_position"])):
                if len(batch["mask_position"][i]) == 0:
                    continue
                t.append(logits[i][batch["mask_position"][i]])
            logits = torch.cat(t)

            labels = torch.cat(batch["mask_label"])
            labels = labels.to(f'cuda:{model.device_ids[0]}')
            
            loss = loss_fn(logits, labels)
            return loss
    
    #validation step
    def validation():
        valid_losses = []
        with torch.no_grad():
            for batch in valid_dataloader:
                inputs = {k: batch[k].to(f"cuda:{model.device_ids[0]}") for k in ["input_ids", "bbox", "pixel_values", "attention_mask"]}
                logits = model.forward(inputs)
                loss = cal_loss(logits, batch)
                valid_losses.append(loss)
            return sum(valid_losses) / len(valid_losses)


    losses = []
    valid_losses = []
    model.train()
    for epoch in range(args.max_epochs):
        for iter, batch in enumerate(train_dataloader):
            inputs = {k: batch[k].to(f"cuda:{model.device_ids[0]}") for k in ["input_ids", "bbox", "pixel_values", "attention_mask"]}
            logits = model.forward(inputs)
            loss = cal_loss(logits, batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if iter % 10 == 0:
                val_loss = validation()
                valid_losses.append(val_loss)
                print(f"{iter}  train_loss: {loss.item()}, valid_loss: {val_loss}", flush=True)
        print("epoch", epoch, loss.item(), flush=True)
    
    torch.save(
    {
        "epoch": args.max_epochs,
        "batch_size": args.batch_size,
        "train_loss_list": losses,
        "valid_loss_list": valid_losses,
        "model_state_dict": model.module.to("cpu").state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    f"{args.output_model_dir}{args.output_file_name}",
    )
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_vocab_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--model_params", type=str)
    parser.add_argument("--output_model_dir", type=str, required=True)
    parser.add_argument("--output_file_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ratio_train", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=1)
    args = parser.parse_args()
    main(args)