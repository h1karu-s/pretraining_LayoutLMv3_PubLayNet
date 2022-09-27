
import collections

import sys
import fitz
import numpy as np
import itertools
import numpy as np
import torch

##debag
import requests

TOKEN = "xoxb-4034955631811-4035078754066-Hc3p7yqYsWNWN6VpGls28DMD"
CHANNEL = "general"
url = "https://slack.com/api/chat.postMessage"
headers = {"Authorization": "Bearer "+TOKEN}

def notification_slack(mes):
    data = {
      "channel": CHANNEL,
      "text": mes
    }
    requests.post(url, headers=headers, data=data)
###########

#create vocab list
# tokenizer = LayoutLMv3Tokenizer("../model/tokenizer_vocab/vocab.json", "../model/tokenizer_vocab/merges.txt")
# ids = range(tokenizer.vocab_size)
# vocab = tokenizer.convert_ids_to_tokens(ids)

MaskedLMInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

#extract text and bbox from  pdfs , Capital word covert into lower words
def extraction_text_from_pdf(file_path, file_names):
    words = []
    bboxes = []
    for i,file_name in enumerate(file_names):
        words.append([])
        bboxes.append([])
        doc = fitz.open(f"{file_path}{file_name}.pdf")
        for page in doc:
            data = page.get_text("words")
            width = page.rect.width
            height = page.rect.height
            for d in data:
                words[i].append(d[4].lower())
                bbox = normalize_bbox(list(d[0:4]), [width, height])
                bboxes[i].append(bbox)      
    return words, bboxes

#for input ids
# add special toknes ( [SEP])
def add_sep_tokens(ids, bboxes, vocab):
    assert len(ids) == len(bboxes), "tokens size isn't  equal to bboxes size." 
    sep_id = vocab.index("</s>")
    new_bboxes = []
    new_ids = []
    #add [sep]
    for i in range(len(ids) -1):
        new_ids.append(ids[i])
        new_bboxes.append(bboxes[i])
        if ids[i] == vocab.index(".") and vocab[ids[i+1]].startswith("Ġ"):
            new_ids.append(sep_id)
            new_bboxes.append([0, 0, 0, 0])
    new_ids.append(ids[-1])
    new_bboxes.append(bboxes[-1])
    return new_ids, new_bboxes

## subset_tokens_from_document内で使用
def padding_attention_mask(input_ids, bboxes, vocab, max_length=512):
    pad_length = max_length - len(input_ids)
    attention_mask = [1] * len(input_ids) + [0]*pad_length
    if len(input_ids) < 512:
        input_ids = input_ids + [vocab.index('<pad>')] * (max_length - len(input_ids))
        bboxes = bboxes + [[0, 0, 0, 0]] * (max_length - len(bboxes))
    return input_ids, bboxes, attention_mask

#長いテキストを入力サイズに分割 + attention_maskを作成+ padding + 分割した文の先頭に<CLS>を追加
def subset_tokens_from_document(tokens_list, bboxes_list, pixel_values, vocab,  max_len = 512):
    subset_tokens_list = []
    subset_bboxes_list = []
    pixel_values_list = []
    attention_mask_list = []
    max_len = max_len - 2 #<cls>, <sep>tokenを追加するために2へらす   
    
    def add_list(tokens, bboxes, pixel):
        #はじめに<CLS>を追加する
        cls_id = vocab.index("<s>")
        tokens.insert(0, cls_id)
        bboxes.insert(0, [0, 0, 0, 0])
        #最後に<SEP>を追加
        sep_id = vocab.index("</s>")
        tokens.append(sep_id)
        bboxes.append([0, 0, 0, 0])
        #attention_maskを作成
        tokens, bboxes, attention_mask = padding_attention_mask(tokens, bboxes, max_len+2)
        
        subset_tokens_list.append(tokens)
        subset_bboxes_list.append(bboxes)
        pixel_values_list.append(pixel)
        attention_mask_list.append(attention_mask)     
        
    for tokens, bboxes, pixel_value in zip(tokens_list, bboxes_list, pixel_values):
        if len(tokens) <= max_len:
            #add attention mask
            
            add_list(tokens, bboxes, pixel_value)
            
        if len(tokens) > max_len:
            div_num = len(tokens) // max_len
            
            if len(tokens) != len(bboxes):
                print("Not equely", len(tokens), len(bboxes))
                print(tokens)
                break
            
            for i  in range(div_num):
                subset_tokens = tokens[i*(max_len): max_len*(i+1)]
                subset_bboxes = bboxes[i*(max_len): max_len*(i+1)]       
        
                add_list(subset_tokens, subset_bboxes, pixel_value)
                
            subset_tokens = tokens[max_len*(i+1):]
            subset_bboxes = bboxes[max_len*(i+1):]
            
            add_list(subset_tokens, subset_bboxes, pixel_value)
    
    return (subset_tokens_list, subset_bboxes_list, pixel_values_list, attention_mask_list)

#token id から span maskをする
#bpe baseではなく word base
def create_span_mask_for_ids(token_ids, masked_lm_prob, max_predictions_per_seq, vocab_words, param , rng):
    cand_indexes = []
    for i, id in enumerate(token_ids):
        if id == vocab_words.index("<s>") or id == vocab_words.index("</s>") or id == vocab_words.index("<pad>"):
            continue

        if len(cand_indexes) >= 1 and not vocab_words[id].startswith("Ġ"):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
    output_tokens = list(token_ids)
    #全単語×0.3(masked_lm_prob)がmaskの対象
    num_to_predict = min(max_predictions_per_seq, 
                      max(1, int(round(len(cand_indexes) * masked_lm_prob))))
    

    span_count = 0
    covered_indexes = [] #mask候補のリスト
    covered_set = set()  # 被らないか確かめるための集合
    #spanのword数が全words数の30%を超えたら終了
    while (span_count < num_to_predict):

        span_length = np.random.poisson(lam=param)
        if span_count + span_length > num_to_predict or span_length == 0:
            continue
        #cand_indexesから初めの単語を決める
        if len(cand_indexes) -(1 + span_length) <= 0:
            break
            # continue
        start_index = rng.randint(0, len(cand_indexes)-(1 + span_length))
        #span_lengthからsubword単位のspanの範囲を決める
        covered_index = cand_indexes[start_index: start_index +span_length]
        covered_index = list(itertools.chain.from_iterable(covered_index))
        if covered_set.isdisjoint(set(covered_index)):
            covered_set = covered_set | set(covered_index)
            span_count += span_length
            # print(span_length)
            covered_indexes.append(covered_index)
            # print(covered_indexes)

    masked_lms = []
    for span_index in covered_indexes:
        if rng.random() < 0.8:
            mask_token_id = vocab_words.index("<mask>")
            masked_tokens= [mask_token_id for _ in range(len(span_index))]
            #maskした場所と元のtokenを記録
            for i in span_index:
                masked_lms.append(MaskedLMInstance(index=i, label=token_ids[i]))

        else:
            if rng.random() < 0.5:
                masked_tokens = [token_ids[i] for i in span_index]

            else:
                #replace words
                masked_tokens = [rng.randint(0, len(vocab_words) - 1) for _ in range(len(span_index))]
                
        for i, index in enumerate(span_index):
            output_tokens[index] = masked_tokens[i]

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []    
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    
    #debag
    # if len(token_ids) > 300 and len(masked_lm_positions) < 2:
    #     print(f"error!!! token length: {len(token_ids)}, postions : {masked_lm_positions}, num_to_predict:{num_to_predict}, span_coont:{span_count},covered_indexes:{covered_indexes}, cand_index:{len(cand_indexes)} coverd_indexes_lenght:{len(covered_indexes)}", flush=True)


    return (output_tokens, masked_lm_positions, masked_lm_labels)

def batch_create_span_mask(input_ids_tensor, masked_lm_prob, max_predictions_per_seq, vocab_words, param , rng):
    b_output_tokens = []
    b_masked_lm_positions = []
    b_masked_lm_labels = []
    input_ids_list = input_ids_tensor.tolist()
    for input_ids in input_ids_list:
        output_tokens, masked_lm_positions, masked_lm_labels = create_span_mask_for_ids(
            input_ids, masked_lm_prob, max_predictions_per_seq, vocab_words, param, rng)
        
        b_output_tokens.append(output_tokens)
        b_masked_lm_positions.append(torch.tensor(masked_lm_positions))
        b_masked_lm_labels.append(torch.tensor(masked_lm_labels))
    # print(len(b_output_tokens[2]), len(b_output_tokens[1]), len(b_output_tokens[0]), len(b_masked_lm_positions[0]))
    return (torch.tensor(b_output_tokens), b_masked_lm_positions, b_masked_lm_labels)

def normalize_bbox(bbox, size):
    x0 = int(1000*bbox[0] / size[0])
    y0 = int(1000*bbox[1] /size[1])
    x1 = int(1000*bbox[2] / size[0])
    y1 = int(1000*bbox[3] / size[1])
    if x0 > 1000: x0 = 1000
    if y0 > 1000: y0 = 1000
    if x1 > 1000: x1 = 1000
    if y1 > 1000: y1 = 1000
    if x0 < 0: x0 = 0
    if y0 < 0: y0 = 0
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    return [x0, y0, x1, y1]
    # return [
    #     int(1000 * bbox[0] / size[0]),
    #     int(1000 * bbox[1] / size[1]),
    #     int(1000 * bbox[2] / size[0]),
    #     int(1000 * bbox[3] / size[1]),
    # ]
        
#軽量版
#長いテキストを入力サイズに分割 + 分割した文の先頭に<CLS>を追加
def subset_tokens_from_document_light(tokens_list, bboxes_list, vocab,  max_len = 512):
    subset_tokens_list = []
    subset_bboxes_list = []
    document_ids = []
    max_len = max_len - 2 #<cls>, <sep>tokenを追加するために2へらす
    
    
    def add_list(tokens, bboxes, id):
        #はじめに<CLS>を追加する
        cls_id = vocab.index("<s>")
        tokens.insert(0, cls_id)
        bboxes.insert(0, [0, 0, 0, 0])
        #最後に<SEP>を追加
        sep_id = vocab.index("</s>")
        tokens.append(sep_id)
        bboxes.append([0, 0, 0, 0])
        subset_tokens_list.append(tokens)
        subset_bboxes_list.append(bboxes)
        document_ids.append(id)        
        
    for doc_id, (tokens, bboxes)  in enumerate(zip(tokens_list, bboxes_list)):
        if len(tokens) <= max_len:
            #add attention mask       
            add_list(tokens, bboxes, doc_id)   
        if len(tokens) > max_len:
            div_num = len(tokens) // max_len         
            if len(tokens) != len(bboxes):
                print("Not equely", len(tokens), len(bboxes))
                print(tokens)
                break           
            for i  in range(div_num):
                subset_tokens = tokens[i*(max_len): max_len*(i+1)]
                subset_bboxes = bboxes[i*(max_len): max_len*(i+1)]              
                add_list(subset_tokens, subset_bboxes, doc_id)
            subset_tokens = tokens[max_len*(i+1):]
            subset_bboxes = bboxes[max_len*(i+1):]         
            add_list(subset_tokens, subset_bboxes, doc_id)
    return (subset_tokens_list, subset_bboxes_list, document_ids)

def init_visual_bbox(img_size=(14, 14), max_len=1000):
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
def create_alignment_label(visual_bbox, text_bbox, bool_mi_pos):
    num_text = len(text_bbox)
    labels = torch.ones(num_text)
    for v_b in visual_bbox[bool_mi_pos]:
        for j, t_b in enumerate(text_bbox):
            if is_content_bbox(t_b, v_b) or is_content_bbox_2(t_b, v_b):
                labels[j] = 0
    alignment_label = labels.to(torch.bool)
    return alignment_label

# (x0, y0, x1, y1) x0, y0比較
def is_content_bbox(text_bbox, image_bbox):
    if (text_bbox[0] >= image_bbox[0] and text_bbox[1] >= image_bbox[1] 
    and text_bbox[0] <= image_bbox[2] and text_bbox[1] <= image_bbox[3]):
        return True
    else:
        return False
    
# (x0, y0, x1, y1) x1, y1比較
def is_content_bbox_2(text_bbox, image_bbox):
    if (text_bbox[2] >= image_bbox[0] and text_bbox[3] >= image_bbox[1] 
    and text_bbox[2] <= image_bbox[2] and text_bbox[3] <= image_bbox[3]):
        return True
    else:
        return False