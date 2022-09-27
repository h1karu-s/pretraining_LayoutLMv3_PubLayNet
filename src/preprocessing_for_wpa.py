# -*-coding:utf-8-*-

import argparse
from cProfile import label
from logging import raiseExceptions
import os
from posixpath import split
from PIL import Image
import pickle

import torch
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer
from torchvision import transforms
from dall_e.utils import map_pixels
from dall_e import load_model

from utils import utils, masking_generator
from utils.slack import notification_slack


window_size = (14, 14)
num_masking_patches = 75
max_mask_patches_per_block = None
min_mask_patches_per_block = 16

# generating mask for the corresponding image
mask_generator = masking_generator.MaskingGenerator(
            window_size, num_masking_patches=num_masking_patches,
            max_num_patches=max_mask_patches_per_block,
            min_num_patches=min_mask_patches_per_block,
        )


def main(args):
    print(args, flush=True)
    device = torch.device('cpu')
    encoder = load_model("https://cdn.openai.com/dall-e/encoder.pkl", device)

    tokenizer = LayoutLMv3Tokenizer(f"{args.tokenizer_vocab_dir}vocab.json", f"{args.tokenizer_vocab_dir}merges.txt")
    ids = range(tokenizer.vocab_size)
    vocab = tokenizer.convert_ids_to_tokens(ids)

    visual_bbox = utils.init_visual_bbox()

    image_names = os.listdir(args.image_file_dir)
    if hasattr(args, 'datasize'):
        image_names = image_names[:args.datasize]
    file_names = []
    for n in image_names:
        name = os.path.splitext(n)[0]
        file_names.append(name)

    print(len(file_names),  flush=True)

    for iter in range(0, len(file_names), args.split_size):
        split_file_names = file_names[iter : iter + args.split_size]
        notification_slack(f"{args.output_filename}: split_file{iter}. file_size is {len(split_file_names)}.")
        print(f"start! extraction words and bboxes from pdf. length is {len(split_file_names)}", flush=True)
        words, bboxes = utils.extraction_text_from_pdf(args.pdf_file_dir, split_file_names)
        enc_input_ids = []
        enc_bboxes = []
        for i in range(len(words)):
            try:
                enc = tokenizer(text=words[i], boxes = bboxes[i], add_special_tokens=False)
                enc_input_ids.append(enc["input_ids"])
                enc_bboxes.append(enc["bbox"])
            except Exception as e:
                print(split_file_names.pop(i), e, flush=True)
        notification_slack(f"{args.output_filename}: finish tokenize.")
        #Original image (resized + normalized): pixel_values
        #Image prepared for DALL-E encoder (map_pixels): pixel_values_dall_e
        pixel_values = []
        labels_list = []
        bool_masked_pos_list = []
        feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
        for file_name in split_file_names:
            image = Image.open(f"{args.image_file_dir}{file_name}.png")
            #pixel_values
            pixel_value = feature_extractor(image)["pixel_values"]
            pixel_values.append(pixel_value)
            #pixel_values_dall_e
            visual_token_transform = transforms.Compose([
                    transforms.Resize((112,112), transforms.InterpolationMode.LANCZOS),
                    transforms.ToTensor(),
                ])
            pixel_values_dall_e = visual_token_transform(image).unsqueeze(0)
            pixel_values_dall_e = map_pixels(pixel_values_dall_e)
            with torch.no_grad():
                z_logits = encoder(pixel_values_dall_e)
                input_ids = torch.argmax(z_logits, axis=1).flatten(1)
                #create mask position
                bool_masked_pos = mask_generator()
                bool_masked_pos = torch.from_numpy(bool_masked_pos).unsqueeze(0)
                bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
                labels = input_ids[bool_masked_pos]
                labels_list.append(labels)
                bool_masked_pos_list.append(bool_masked_pos)

        notification_slack(f"{args.output_filename}: start create subset token!.")
        print("start! create subset tokens", flush=True)
        tokens, bboxes, doc_ids = utils.subset_tokens_from_document_light(enc_input_ids, enc_bboxes, vocab, max_len=512)
        notification_slack(f"{args.output_filename}: fiish create subset token! and createing dataset.")
        dataset = []
        for i in range(len(tokens)):
            al_labels = utils.create_alignment_label(
                visual_bbox=visual_bbox,
                text_bbox=bboxes[i],
                bool_mi_pos=bool_masked_pos_list[doc_ids[i]][0],
                )
            dataset.append({"input_ids": tokens[i], 
            "bbox": bboxes[i], 
            "pixel_values": pixel_values[doc_ids[i]][0], 
            "label": labels_list[doc_ids[i]],
            "bool_masked_pos": bool_masked_pos_list[doc_ids[i]][0],
            "alignment_labels": al_labels
            })
        notification_slack(f"{args.output_filename}: start saving....")

        with open(f"{args.output_dir}{args.output_filename}/{iter}.pkl", 'wb') as f:
            pickle.dump(dataset, f, protocol=5)
        notification_slack(f"{args.output_filename}: saved: {iter}.pkl")
    
    notification_slack(f"{args.output_filename}: finish all process!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_vocab_dir", type=str)
    parser.add_argument("--image_file_dir", type=str, required=True)
    parser.add_argument("--pdf_file_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    parser.add_argument("--datasize", type=int)
    parser.add_argument("--split_size", type=int)
    args = parser.parse_args()
    main(args)