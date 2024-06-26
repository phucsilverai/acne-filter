import sys
sys.path.append('/home/phucle/code/acne-filter/LLaVA')

import os
import requests
from io import BytesIO

from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch

from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

class LLaVA():
    def __init__(self, model_path="4bit/llava-v1.5-13b-3GB"):
        kwargs = {"device_map": "auto"}
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )

        self.model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device='cuda')
        self.image_processor = vision_tower.image_processor

    def caption_image(self, image_file, prompt):
        if type(image_file) == str:
            if image_file.startswith('http') or image_file.startswith('https'):
                response = requests.get(image_file)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_file).convert('RGB')
        else:
            image = image_file
            
        disable_torch_init()
        conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        inp = f"{roles[0]}: {prompt}"
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        raw_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(raw_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2, 
                                        max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
            outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs
            output = outputs.rsplit('</s>', 1)[0]
        return image, output