# -*-coding:utf-8-*-

import argparse
from cProfile import label
import os
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


def main(args):
    print(args, flush=True)
    device = torch.device('cpu')
    encoder = load_model("https://cdn.openai.com/dall-e/encoder.pkl", device)

    tokenizer = LayoutLMv3Tokenizer(f"{args.tokenizer_vocab_dir}vocab.json", f"{args.tokenizer_vocab_dir}merges.txt")
    ids = range(tokenizer.vocab_size)
    vocab = tokenizer.convert_ids_to_tokens(ids)

    image_names = os.listdir(args.image_file_dir)
    if hasattr(args, 'datasize'):
        image_names = image_names[:args.datasize]
    file_names = []
    for n in image_names:
        name = os.path.splitext(n)[0]
        file_names.append(name)

    print("start! extraction words and bboxes from pdf.", flush=True)
    words, bboxes = utils.extraction_text_from_pdf(args.pdf_file_dir, file_names)

    enc_input_ids = []
    enc_bboxes = []
    for i in range(len(words)):
        try:
            enc = tokenizer(text=words[i], boxes = bboxes[i], add_special_tokens=False)
            enc_input_ids.append(enc["input_ids"])
            enc_bboxes.append(enc["bbox"])
        except Exception as e:
            print(file_names.pop(i), e, flush=True)
    
    #Original image (resized + normalized): pixel_values
    #Image prepared for DALL-E encoder (map_pixels): pixel_values_dall_e
    pixel_values = []
    labels_list = []
    bool_masked_pos_list = []
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
    for file_name in file_names:
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

    print("start! create subset tokens", flush=True)
    tokens, bboxes, doc_ids = utils.subset_tokens_from_document_light(enc_input_ids, enc_bboxes, vocab, max_len=512)

    dataset = []
    for i in range(len(tokens)):
        dataset.append({"input_ids": tokens[i], 
        "bbox": bboxes[i], 
        "pixel_values": pixel_values[doc_ids[i]][0], 
        "label": labels_list[doc_ids[i]],
        "bool_masked_pos": bool_masked_pos_list[doc_ids[i]]})
    
    with open(f"{args.output_dir}{args.output_filename}", 'wb') as f:
        pickle.dump(dataset, f, protocol=4)
    
    notification_slack("finish!!!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_vocab_dir", type=str)
    parser.add_argument("--image_file_dir", type=str, required=True)
    parser.add_argument("--pdf_file_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    parser.add_argument("--datasize", type=int)
    args = parser.parse_args()
    main(args)