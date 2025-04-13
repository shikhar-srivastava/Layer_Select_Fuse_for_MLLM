#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
                         
from .modeling_llama import LlamaModel, LlamaForCausalLM
from .configuration_llama import LlamaConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.mm_utils import unbiased_cka

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        images_features: Optional[List[torch.FloatTensor]] = None,
        image_token_mask: Optional[torch.FloatTensor] = None,
        compute_cka: bool = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # if compute_cka:
        # Then compare image_features 
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        original_output_hidden_states = output_hidden_states
        output_hidden_states = True if compute_cka else (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_features_list = None
        image_features_f = None
        if "E" in self.config.layer_fusing_strategy:
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    image_features_f,
                    image_token_mask,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
                )
            if images_features is None:
                images_features = image_features_f
        else:
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    image_features_list,
                    image_features_f,
                    image_token_mask,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
                )


            if images_features is None and image_features_list is not None:
                images_features = image_features_list + [image_features_f]



        outputs = self.model(
            image_token_mask=image_token_mask,
            images_features=images_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # TODO: 
        # 1. CKA for each batch separately, since it's not meaningful to compute CKA across unaligned (image, prompt) pairs
        # 2. LLM input should be text only, no image ; when computing CKA. 
        #    - dummy image tokens would change (text) hidden states.
        # NOTE:
        # Currently it's comparing input image embeddings to hidden state image embeddings from each layer (which is not what we want).
        # 
    
        cka_similarities = None
        if compute_cka and outputs.hidden_states is not None and image_token_mask is not None:
            cka_similarities = {}
            image_token_mask_bool = image_token_mask.bool()
            if image_token_mask_bool.any(): 
                image_input_embeds = inputs_embeds[image_token_mask_bool] # [bsz, n_image_patches, d_model]
                image_input_embeds = image_input_embeds.view(-1, image_input_embeds.size(-1)) # [bsz * n_image_patches, d_model]
                for layer_idx, layer_hidden_state in enumerate(outputs.hidden_states):
                    layer_image_hidden_states = layer_hidden_state[image_token_mask_bool]
                    layer_image_hidden_states = layer_image_hidden_states.view(-1, layer_image_hidden_states.size(-1))
                    if image_input_embeds.shape[0] == layer_image_hidden_states.shape[0] and image_input_embeds.shape[0] > 1:
                        try:
                            cka_val = unbiased_cka(image_input_embeds, layer_image_hidden_states)
                            cka_similarities[f'layer_{layer_idx}'] = cka_val.item()
                        except Exception as e:
                            print(f"Warning: CKA computation failed for layer {layer_idx}: {e}")
                            cka_similarities[f'layer_{layer_idx}'] = None
                    elif image_input_embeds.shape[0] <= 1:
                        cka_similarities[f'layer_{layer_idx}'] = None
                    else:
                        print(f"Warning: Shape mismatch for CKA in layer {layer_idx}. Input: {image_input_embeds.shape}, Layer: {layer_image_hidden_states.shape}. Skipping.")
                        cka_similarities[f'layer_{layer_idx}'] = None
            else:
                num_layers = len(outputs.hidden_states) if outputs.hidden_states else 0
                for layer_idx in range(num_layers):
                     cka_similarities[f'layer_{layer_idx}'] = None

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            output = output + (cka_similarities,)
            if not original_output_hidden_states and hasattr(outputs, 'hidden_states'):
                 print("Warning: Requesting CKA without return_dict=True might lead to unexpected output tuple structure.")
                 pass
            return (loss,) + output if loss is not None else output

        output_obj = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if (original_output_hidden_states or compute_cka) else None,
            attentions=outputs.attentions,
        )
        if cka_similarities is not None:
            output_obj.cka_similarities = cka_similarities
            
        return output_obj


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        image_features_list = None

        if "E" in self.config.layer_fusing_strategy:
            if images is not None:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _,
                    image_features_f,
                    image_token_mask
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes
                )
            else:
                inputs_embeds = self.get_model().embed_tokens(inputs)

            if image_features_f != None:
                images_features = image_features_f
        else:
            if images is not None:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _,
                    image_features_list,
                    image_features_f,
                    image_token_mask
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes
                )
            else:
                inputs_embeds = self.get_model().embed_tokens(inputs)
            if image_features_list != None:
                images_features = image_features_list + [image_features_f]



        return super().generate(
            image_token_mask = image_token_mask,
            images_features = images_features,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    



    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_token_mask = kwargs.pop("image_token_mask", None)
        images_features = kwargs.pop("images_features", None)


        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes

        if image_token_mask is not None:
            inputs['image_token_mask'] = image_token_mask
        if images_features is not None:
            inputs['images_features'] = images_features


        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
