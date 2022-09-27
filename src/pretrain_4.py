# -*-coding:utf-8-*-

import argparse
import os
import  pickle5 as pickle
import math

import torch
import matplotlib.pyplot as plt
from transformers import AutoConfig, RobertaModel, LayoutLMv3Tokenizer
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup

from model import  My_DataLoader
from model.LayoutLMv3forMIM import LayoutLMv3ForPretraining
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
    notification_slack("start")
    if not torch.cuda.is_available():
        raise ValueError("GPU is not available.")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device_ids = list(range(torch.cuda.device_count()))

    tokenizer = LayoutLMv3Tokenizer(f"{args.tokenizer_vocab_dir}vocab.json", f"{args.tokenizer_vocab_dir}merges.txt")
    ids = range(tokenizer.vocab_size)
    vocab = tokenizer.convert_ids_to_tokens(ids)

    save_hparams(args)
    
    if not args.model_params is None:
        model = torch.load(args.model_params)
    else:
        config = AutoConfig.from_pretrained(args.model_name)
        config.num_visual_tokens = 8192
        model = LayoutLMv3ForPretraining(config)
        Roberta_model = RobertaModel.from_pretrained("roberta-base")
        ## embedidng 層の重みをRobertaの重みで初期化
        weight_size = model.state_dict()["model.embeddings.word_embeddings.weight"].shape
        for i in range(weight_size[0]):
          model.state_dict()["model.embeddings.word_embeddings.weight"][i] = \
          Roberta_model.state_dict()["embeddings.word_embeddings.weight"][i]
    #optimizer 
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2, betas=(0.9, 0.98))
    #cross entropy
    criterion = torch.nn.CrossEntropyLoss()
    #modelをGPUへ
    model = torch.nn.DataParallel(model, device_ids = device_ids)
    model = model.to(f'cuda:{model.device_ids[0]}')
    #load input_file
    data = []
    input_names = os.listdir(args.input_file)
    if args.datasize is not None:
        input_names = input_names[:args.datasize]
    notification_slack(f"input_file_length: {len(input_names)}")
    for file_name in input_names:
        with open(f"{args.input_file}{file_name}", "rb") as f:
            d = pickle.load(f)
            data += d
    notification_slack(f"pretraing: datasize is {len(data)}")
    #divide into train and valid
    n_train = math.floor(len(data) * args.ratio_train)
    train_data = data[:n_train]
    valid_data = data[n_train:]
    notification_slack(f"pretraing: train_data is {len(train_data)}, valid_data is {len(valid_data)}.")
    #create dataloader
    my_dataloader = My_DataLoader.My_Dataloader(vocab)
    train_dataloader = my_dataloader(train_data, batch_size=args.batch_size, shuffle=False)
    valid_dataloader = my_dataloader(valid_data, batch_size=args.batch_size, shuffle=False)

    #scheduler warm up lineary over fist 0.4% step
    iter_per_epoch = len(train_dataloader)
    num_warmup_steps = round((iter_per_epoch * args.max_epochs) * 0.048)
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
    
    #define caluculation ml?loss
    def cal_ml_loss(text_logits, batch):
        t = []
        for i in range(len(batch["ml_position"])):
            if len(batch["ml_position"][i]) == 0:
                continue
            t.append(text_logits[i][batch["ml_position"][i]])
        if len(t) == 0:
            return 0
        predict_word_token = torch.cat(t)
        labels = torch.cat(batch["ml_label"])
        labels = labels.to(f'cuda:{model.device_ids[0]}')
        loss = criterion(predict_word_token + 1e-12, labels)
        return loss

    #define caluculation ml?loss
    def cal_ml_loss_debag(text_logits, batch, type):
        t = []
        for i in range(len(batch["ml_position"])):
            if len(batch["ml_position"][i]) == 0:
                continue
            t.append(text_logits[i][batch["ml_position"][i]])
        if len(t) == 0:
            notification_slack(f"type: {type}")
            with open(f"./data/preprocessing_shared/error_file.pkl", 'wb') as f:
                pickle.dump(batch, f, protocol=5)
            notification_slack(f"atten:{len(batch['attention_mask'])}")
            return None
        predict_word_token = torch.cat(t)
        labels = torch.cat(batch["ml_label"])
        labels = labels.to(f'cuda:{model.device_ids[0]}')
        loss = criterion(predict_word_token + 1e-12, labels)
        return loss

    #define calculation mi_loss
    def cal_mi_loss(image_logits, batch):
        image_logits = image_logits[:,1:]
        predict_visual_token = image_logits[batch["bool_mi_pos"]].to(torch.float32)
        labels = torch.cat(batch["mi_label"])
        labels = labels.to(f'cuda:{model.device_ids[0]}')
        loss = criterion(predict_visual_token + 1e-12, labels)
        return loss
    
    #define calculation wpa loss
    def cal_wpa_loss(wpa_logits, batch):
        w_logits = wpa_logits[:,:512]
        #padとlanguage maskのindexを除外
        t  = []
        for i in range(wpa_logits.shape[0]):
            bool_index = torch.ones(512)
            bool_index[batch["ml_position"][i]] = 0
            bool_index = bool_index * batch["attention_mask"][i]
            t.append(bool_index)
        bool_indexes = torch.stack(t).to(torch.bool)
        predict_label = w_logits[bool_indexes]
        labels = batch["alignment_labels"][bool_indexes].to(torch.long)
        labels = labels.to(f'cuda:{model.device_ids[0]}')
        loss = criterion(predict_label + 1e-12, labels)
        return loss
    
    #validation step
    def validation(dataloader):
        losses = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: batch[k].to(f"cuda:{model.device_ids[0]}") for k in ["input_ids", "bbox", "pixel_values", "attention_mask", "bool_mi_pos"]}
                text_logits, image_logits, wpa_logits = model.forward(inputs)
                cal_ml_loss_debag(text_logits, batch, "valid")
                mi_loss = cal_mi_loss(image_logits, batch)
                wpa_loss = cal_wpa_loss(wpa_logits, batch)
                try:
                    val_loss = ml_loss + mi_loss + wpa_loss
                except Exception as e:
                    notification_slack(f"ml_loss: {ml_loss}")
                    notification_slack(f"mi_loss: {mi_loss}")
                    notification_slack(f"wpa_loss: {wpa_loss}")
                    notification_slack(f"losses: {losses}")
                    print(e)
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
            inputs = {k: batch[k].to(f"cuda:{model.device_ids[0]}") for k in ["input_ids", "bbox", "pixel_values", "attention_mask", "bool_mi_pos"]}
            text_logits, image_logits, wpa_logits = model.forward(inputs)
            ml_loss = cal_ml_loss_debag(text_logits, batch, "train")
            mi_loss = cal_mi_loss(image_logits, batch)
            wpa_loss = cal_wpa_loss(wpa_logits, batch)

            loss = ml_loss + mi_loss + wpa_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if i % math.floor(iter_per_epoch*0.1) == 0:
                iter_list.append(iter)
                train_losses.append(loss.item())
                val_loss = validation(valid_dataloader)
                valid_losses.append(val_loss)              
                print(f"{iter}  train_loss: {loss.item()}, valid_loss: {val_loss}", flush=True)
        save_loss_epcoh(args, model, epoch, iter_list, train_losses, valid_losses, optimizer)
        print("epoch", epoch, loss.item(), flush=True)
        
    # save_hparams(args)
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
    parser.add_argument("--datasize", type=int)
    args = parser.parse_args()
    main(args)