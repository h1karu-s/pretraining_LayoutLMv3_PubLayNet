# -*-coding:utf-8-*-

import argparse
import os
import  pickle
import math

import torch
import matplotlib.pyplot as plt
from transformers import AutoConfig, RobertaModel, LayoutLMv3Tokenizer
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup

from model import LayoutLMv3forMLM, My_DataLoader
from utils.slack import notification_slack


def plot_graph(args, epoch, iter_list, train_losses, val_losses):
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.plot(iter_list, train_losses)
    plt.plot(iter_list, val_losses)
    plt.legend(["train_loss", "valid_loss"])
    fig.savefig(f"{args.output_model_dir}epoch_{epoch}/loss.png")

def save_hparams(args):
    with open(f"{args.output_model_dir}hparams.txt", mode="w") as f:
        f.writelines(str(args.__dict__))

#save fun 
def save_loss_epcoh(args, model, epoch, iter_list, train_losses, valid_losses, optimizer):
    os.makedirs(f"{args.output_model_dir}epoch_{epoch}", exist_ok = True)
    plot_graph(args, epoch, iter_list, train_losses, valid_losses) 
    torch.save(
    {
        "epoch": epoch,
        "train_loss_list": train_losses,
        "valid_loss_list": valid_losses,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    f"{args.output_model_dir}epoch_{epoch}/checkpoint.cpt",
    )
    notification_slack(f"epoch:{epoch}が終了しました。valid_lossは{valid_losses[-1]}です。")
         

def main(args):
    print(args, flush=True)
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
        if len(t) == 0:
            # print(batch["mask_position"])
            return 
        logits = torch.cat(t)
        labels = torch.cat(batch["mask_label"])
        labels = labels.to(f'cuda:{model.device_ids[0]}')
        loss = loss_fn(logits, labels)
        return loss
    
    #validation step
    def validation():
        losses = []
        with torch.no_grad():
            for batch in valid_dataloader:
                inputs = {k: batch[k].to(f"cuda:{model.device_ids[0]}") for k in ["input_ids", "bbox", "pixel_values", "attention_mask"]}
                logits = model.forward(inputs)
                val_loss = cal_loss(logits, batch)
                if val_loss is None:
                    continue 
                losses.append(val_loss.item())
            if len(losses) == 0:
                print("losses length is 0", flush=True)
                return 
            return sum(losses) / len(losses)

    train_losses = []
    valid_losses = []
    iter_list = []
    iter_per_epoch = len(train_dataloader)
    model.train()
    for epoch in range(args.max_epochs):
        for i, batch in enumerate(train_dataloader):
            iter = epoch * iter_per_epoch + i
            inputs = {k: batch[k].to(f"cuda:{model.device_ids[0]}") for k in ["input_ids", "bbox", "pixel_values", "attention_mask"]}
            logits = model.forward(inputs)
            loss = cal_loss(logits, batch)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if i % math.floor(iter_per_epoch*0.1) == 0:
                iter_list.append(iter)
                train_losses.append(loss.item())
                val_loss = validation()
                valid_losses.append(val_loss)              
                print(f"{iter}  train_loss: {loss.item()}, valid_loss: {val_loss}", flush=True)
        save_loss_epcoh(args, model, epoch, iter_list, train_losses, valid_losses, optimizer)
        print("epoch", epoch, loss.item(), flush=True)
        
    save_hparams(args)
    notification_slack("学習が無事に終わりました。")
     

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
    parser.add_argument("--max_epochs", type=int, default=2)
    args = parser.parse_args()
    main(args)