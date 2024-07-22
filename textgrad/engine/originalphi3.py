import os
import platformdirs
from PIL import Image
import requests


from peft import PeftConfig, PeftModel
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline,
    AutoProcessor
)

from .base import EngineLM, CachedEngine
from PIL import Image


class originalChatphi3(EngineLM, CachedEngine):
    def __init__(
            self, 
            model_name = "microsoft/Phi-3-vision-128k-instruct", 
            processor_name = "microsoft/Phi-3-vision-128k-instruct",
            max_seq_length=2048, 
            dtype='auto', 
            load_in_4bit=True,
            HF_TOKEN=None):
        
        self.model_name = model_name
        self.processor_name = processor_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.HF_TOKEN = HF_TOKEN
        # Initialize paths and processor
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_together_{model_name}.db")
        
        super().__init__(cache_path=cache_path)






        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype='auto',
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
        attn_implementation='flash_attention_2',token=HF_TOKEN

        ) 

        self.processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
            
    def generate(self, question, image_path=None, system_prompt = None, eos_token_id=None ):
        
        eos_token_id = self.processor.tokenizer.eos_token_id

        image_path = None
        text_content = None

        # Process the list
        for item in question:
            if item.startswith('image:'):
                path = item[len('image: '):]  # Extract the path after 'image: '
            elif item.startswith('text:'):
                text_content = item[len('text: '):]  # Extract the text after 'text: '
        messages = [ 
            {"role": "user", "content": f"<|image_1|>\n{system_prompt}"}, 
            {"role": "user", "content": f"{text_content}"} ]
        if path != None:    
            # image = Image.open(requests.get(image_path.value, stream=True).raw)
            image = Image.open(path)
            image.show()
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"##############path{path}")
        if prompt.endswith("<|endoftext|>"):
            prompt = prompt.rstrip("<|endoftext|>")
        
        print(f"#####################DEBUG_prompt: {prompt}")
        if image_path != None: 
            inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0")
        else:
            inputs = self.processor(prompt, images=None, return_tensors="pt").to("cuda:0")

        generation_args = { "max_new_tokens": 500, "temperature": 0.0, "do_sample": False }
        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        print (f"################DEBUG response{response}")
        return response

            


    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)