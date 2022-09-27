# -*-coding:utf-8-*-

import argparse
import os
import  pickle5 as pickle
import math
import contextlib
import random 

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoConfig, RobertaModel, LayoutLMv3Tokenizer
from torch.optim import AdamW
from transformers import get_constant_schedule_with_warmup

from model import  My_DataLoader
from model.LayoutLMv3forMIM import LayoutLMv3ForPretraining
from utils.slack import notification_slack

#再現性
seed = 3407
def fix_seed(seed):
    # random
    random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@contextlib.contextmanager
def temp_np_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

@contextlib.contextmanager
def temp_random_seed(seed):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)

def plot_graph(args, epoch, iter_list, train_losses, val_losses):
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.plot(iter_list, train_losses)
    plt.plot(iter_list, val_losses)
    plt.legend(["train_loss", "valid_loss"])
    fig.savefig(f"{args.output_model_dir}epoch_{epoch}/loss.png")

def plot_graph_2(args, epoch, iter_list, mi_losses, ml_losses, wpa_losses):
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.plot(iter_list, mi_losses)
    plt.plot(iter_list, ml_losses)
    plt.plot(iter_list, wpa_losses)
    plt.legend(["ML", "MI", "WPA"])
    fig.savefig(f"{args.output_model_dir}epoch_{epoch}/indiv_loss.png")

def plot_graph_3(args, epoch, iter_list, accesML, accesMI):
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.xlabel("iter")
    plt.ylabel("acc")
    plt.plot(iter_list, accesML)
    plt.plot(iter_list, accesMI)
    plt.legend(["ML", "MI"])
    fig.savefig(f"{args.output_model_dir}epoch_{epoch}/acces.png")

def save_hparams(args):
    with open(f"{args.output_model_dir}hparams.txt", mode="w") as f:
        f.writelines(str(args.__dict__))

#save fun 
def save_loss_epcoh(args, model, epoch, iter_list, train_losses, valid_losses, \
    ml_losses, mi_losses, wpa_losses, accesML, accesMI, optimizer, scheduler):
    #os.makedirs pyenv cannot change working directory to .... ↓
    # os.makedirs(f"{args.output_model_dir}epoch_{epoch}", exist_ok = True)
    plot_graph(args, epoch, iter_list, train_losses, valid_losses) 
    plot_graph_2(args, epoch, iter_list, ml_losses, mi_losses, wpa_losses)
    plot_graph_3(args, epoch, iter_list, accesML, accesMI)
    torch.save(
    {
        "epoch": epoch,
        "iter_list": iter_list,
        "train_loss_list": train_losses,
        "valid_loss_list": valid_losses,
        "ml_losses": ml_losses,
        "mi_losses": mi_losses,
        "wpa_losses": wpa_losses,
        "accesML": accesML,
        "accesMI": accesMI,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
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

    save_hparams(args)
    
    if not args.model_params is None:
        checkpoint = torch.load(args.model_params, map_location=torch.device('cpu'))
        config = AutoConfig.from_pretrained(args.model_name)
        config.num_visual_tokens = 8192
        model = LayoutLMv3ForPretraining(config)
        model.load_state_dict(checkpoint["model_state_dict"])
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
    
    #modelをGPUへ
    model = torch.nn.DataParallel(model, device_ids = device_ids)
    model = model.to(f'cuda:{model.device_ids[0]}')

    #optimizer 
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2, betas=(0.9, 0.98))
    if not args.model_params is None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #cross entropy
    criterion = torch.nn.CrossEntropyLoss()

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
    my_dataloader = My_DataLoader.My_Dataloader(vocab, random)
    train_dataloader = my_dataloader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = my_dataloader(valid_data, batch_size=args.batch_size, shuffle=False)

    #scheduler warm up lineary over fist 0.4% step
    iter_per_epoch = len(train_dataloader)
    num_warmup_steps = round((iter_per_epoch * args.max_epochs) * 0.048)
    if not args.model_params is None:
        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("not scheduler", flush = True)
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
    
    #define caluculation ml?loss
    def cal_ml_loss(text_logits, batch):
        t = []
        for i in range(len(batch["ml_position"])):
            if len(batch["ml_position"][i]) == 0:
                continue
            t.append(text_logits[i][batch["ml_position"][i]])
        if len(t) == 0:
            notification_slack("pretrain_3.py: len(t)==0")
            return 0
        t_logits = torch.cat(t)
        labels = torch.cat(batch["ml_label"])
        labels = labels.to(f'cuda:{model.device_ids[0]}')
        loss = criterion(t_logits+ 1e-12, labels)
        accML = (t_logits.argmax(-1) == labels).sum() / len(labels)
        return loss, accML

    #define calculation mi_loss
    def cal_mi_loss(image_logits, batch):
        image_logits = image_logits[:,1:]
        if (image_logits.shape[0] != batch["bool_mi_pos"].shape[0] or image_logits.shape[1] != batch["bool_mi_pos"].shape[1]):
            notification_slack(f"diff imaeg_logit.shape and bool_mi_pos shape{image_logits.shape}, {batch['bool_mi_pos'].shape}")
            return 0
        predict_visual_token = image_logits[batch["bool_mi_pos"]].to(torch.float32)
        labels = torch.cat(batch["mi_label"])
        labels = labels.to(f'cuda:{model.device_ids[0]}')
        loss = criterion(predict_visual_token + 1e-12, labels)
        accMI = (predict_visual_token.argmax(-1) == labels).sum() / len(labels)
        return loss, accMI
    
    
    
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
    def validation():
        losses = []
        ml_losses = []
        mi_losses = []
        wpa_losses = []
        accesML = []
        accesMI = []
        with torch.no_grad():
            with temp_np_seed(3407):
                with temp_random_seed(3407):
                    for batch in valid_dataloader:
                        inputs = {k: batch[k].to(f"cuda:{model.device_ids[0]}") for k in ["input_ids", "bbox", "pixel_values", "attention_mask", "bool_mi_pos"]}
                        text_logits, image_logits, wpa_logits = model.forward(inputs)
                        ml_loss, accML = cal_ml_loss(text_logits, batch)
                        mi_loss, accMI = cal_mi_loss(image_logits, batch)
                        wpa_loss = cal_wpa_loss(wpa_logits, batch)
                        val_loss = ml_loss + mi_loss + wpa_loss
                        losses.append(val_loss.item())
                        ml_losses.append(ml_loss.item())
                        mi_losses.append(mi_loss.item())
                        wpa_losses.append(wpa_loss.item())
                        accesML.append(accML.item())
                        accesMI.append(accMI.item())
                        ave_losses = sum(losses) / len(losses)
                        ave_ml = sum(ml_losses) / len(ml_losses)
                        ave_mi =  sum(mi_losses) / len(mi_losses)
                        ave_wpa = sum(wpa_losses) / len(wpa_losses)
                        ave_accML = sum(accesML) / len(accesML)
                        ave_accMI = sum(accesMI) / len(accesMI)
                    return ave_losses, (ave_ml, ave_mi, ave_wpa), (ave_accML, ave_accMI)
    
    train_losses = []
    valid_losses = []
    ml_losses = []
    mi_losses = []
    wpa_losses = []
    accesML = []
    accesMI = []
    iter_list = []
    ##epcoh
    if not args.model_params is None:
        epochs = range(checkpoint["epoch"] +1, args.max_epochs)
        train_losses = checkpoint["train_loss_list"]
        valid_losses = checkpoint["valid_loss_list"]
        iter_list = checkpoint["iter_list"]
        ml_losses = checkpoint["ml_losses"]
        mi_losses = checkpoint["mi_losses"]
        wpa_losses = checkpoint["wpa_losses"]
        accesML = checkpoint["accesML"]
        accesMI = checkpoint["accesMI"]
        # iter_list = [0, 1314, 2628, 3942, 5265, 6
        #570, 7884, 9198, 10512, 11826, 13140,13141, 14455, 15769, 17083, 18397, 19711, 21025, 22339, 23653, 24967, 26281]
        print(epochs, flush=True)
        print(train_losses, flush=True)
        print(len(iter_list),iter_list, flush=True)
    else:
        epochs = range(args.max_epochs)
    
    notification_slack("start training!")
    iter_per_epoch = len(train_dataloader)
    print("iter: ", epochs[0] * iter_per_epoch, flush=True)
    model.train()
    for epoch in epochs:
        for i, batch in enumerate(train_dataloader):
            iter = epoch * iter_per_epoch + i
            inputs = {k: batch[k].to(f"cuda:{model.device_ids[0]}") for k in ["input_ids", "bbox", "pixel_values", "attention_mask", "bool_mi_pos"]}
            text_logits, image_logits, wpa_logits = model.forward(inputs)
            ml_loss, _ = cal_ml_loss(text_logits, batch)
            mi_loss, _ = cal_mi_loss(image_logits, batch)
            wpa_loss = cal_wpa_loss(wpa_logits, batch)

            loss = ml_loss + mi_loss + wpa_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if i % math.floor(iter_per_epoch*0.1) == 0:
                iter_list.append(iter)
                train_losses.append(loss.item())
                val_loss, indv_loss, val_acc = validation()
                valid_losses.append(val_loss)
                ml_losses.append(indv_loss[0])
                mi_losses.append(indv_loss[1])
                wpa_losses.append(indv_loss[2])   
                accesML.append(val_acc[0])
                accesMI.append(val_acc[1])           
                print(f"{iter}  train_loss: {loss.item()}, valid_loss: {val_loss}", flush=True)
                notification_slack(
                    f"e:{epoch}, iter:{iter}, {i},  train_loss: {loss.item()}, valid_loss: {val_loss}, idiv_loss:{str(indv_loss)}, acc:{str(val_acc)}"
                    )
        save_loss_epcoh(
            args = args,
            model = model,
            epoch = epoch,
            iter_list = iter_list,
            train_losses = train_losses, 
            valid_losses = valid_losses, 
            ml_losses = ml_losses, 
            mi_losses = mi_losses, 
            wpa_losses = wpa_losses,
            accesML = accesML,
            accesMI = accesMI,
            optimizer = optimizer, 
            scheduler = scheduler,
            )
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