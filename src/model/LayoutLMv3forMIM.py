import collections

import torch
from torch import  nn
from transformers import LayoutLMv3Model
from transformers.modeling_outputs import BaseModelOutput


LayoutLMv3ForPretraining_Outputs = \
collections.namedtuple("LayoutLMv3ForPretraining_Outputs",["text_logits", "image_logits", "wpa_logits"])


class LayoutLMv3(LayoutLMv3Model):
    def __init__(self, config):
        super().__init__(config)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, config.hidden_size))
  
    def forward_image(self, pixel_values, bool_mi_pos=None):
        embeddings = self.patch_embed(pixel_values)
        batch_size, seq_len, _ = embeddings.size()

        if bool_mi_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = bool_mi_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        # add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add position embeddings
        if self.pos_embed is not None:
            embeddings = embeddings + self.pos_embed
      


        embeddings = self.pos_drop(embeddings)
        embeddings = self.norm(embeddings)

        return embeddings


    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        bool_mi_pos=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Examples:
        ```python
        >>> from transformers import AutoProcessor, AutoModel
        >>> from datasets import load_dataset
        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]
        >>> encoding = processor(image, words, boxes=boxes, return_tensors="pt")
        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif pixel_values is not None:
            batch_size = len(pixel_values)
            device = pixel_values.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or pixel_values")

        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

        final_bbox = final_position_ids = None
        patch_height = patch_width = None
        if pixel_values is not None:
            patch_height, patch_width = int(pixel_values.shape[2] / self.config.patch_size), int(
                pixel_values.shape[3] / self.config.patch_size
            )
            visual_embeddings = self.forward_image(pixel_values, bool_mi_pos)
            visual_attention_mask = torch.ones(
                (batch_size, visual_embeddings.shape[1]), dtype=torch.long, device=device
            )
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            else:
                attention_mask = visual_attention_mask

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self.calculate_visual_bbox(device, dtype=torch.long, batch_size=batch_size)
                    if bbox is not None:
                        final_bbox = torch.cat([bbox, visual_bbox], dim=1)
                    else:
                        final_bbox = visual_bbox

                visual_position_ids = torch.arange(
                    0, visual_embeddings.shape[1], dtype=torch.long, device=device
                ).repeat(batch_size, 1)
                if input_ids is not None or inputs_embeds is not None:
                    position_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0)
                    position_ids = position_ids.expand(input_shape)
                    final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)
                else:
                    final_position_ids = visual_position_ids

            if input_ids is not None or inputs_embeds is not None:
                embedding_output = torch.cat([embedding_output, visual_embeddings], dim=1)
            else:
                embedding_output = visual_embeddings

            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, None, device, dtype=embedding_output.dtype
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_height=patch_height,
            patch_width=patch_width,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

#model head for Word-Patch Alignment(WPA)
class HeadForWPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, 2, bias = False)
        self.bias = nn.Parameter(torch.zeros(2))
        self.decoder.bias = self.bias
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

## model head for masked image model 
class HeadForMIM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.num_visual_tokens, bias = False)
        self.bias = nn.Parameter(torch.zeros(config.num_visual_tokens))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

## model head for masked language model 
class HeadForMLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias = False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

## layoutLMv3(MIM) + head for MLM 
##config : config = AutoConfig.from_pretrained("microsoft/layoutlmv3-base")
class LayoutLMv3ForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LayoutLMv3(config)
        self.HeadForMLM = HeadForMLM(config)
        self.HeadForMIM = HeadForMIM(config)
        self.HeadForWPA = HeadForWPA(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input):
        if input["input_ids"].shape[1] > 512:
            print("over lengths")
        outputs = self.model(**input)
        outputs = outputs[0]
        text_outputs = outputs[:, :512]
        image_outputs = outputs[:, 512:]
        text_outputs = self.HeadForMLM(text_outputs)
        image_outputs = self.HeadForMIM(image_outputs)
        wpa_outputs = self.HeadForWPA(outputs)

        return LayoutLMv3ForPretraining_Outputs(
            text_logits = text_outputs,
            image_logits = image_outputs,
            wpa_logits = wpa_outputs
        )