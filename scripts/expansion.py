# Fooocus GPT2 Expansion
# Algorithm created by Lvmin Zhang at 2023, Stanford
# If used inside Fooocus, any use is permitted.
# If used outside Fooocus, only non-commercial use is permitted (CC-By NC 4.0).
# This applies to the word list, vocab, model, and algorithm.


import os
import torch
import math
from pathlib import Path

from modules.shared import opts
from transformers.generation.logits_process import LogitsProcessorList
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from model_loader import load_file_from_url


# limitation of np.random.seed(), called from transformers.set_seed()
SEED_LIMIT_NUMPY = 2**32
neg_inf = - 8192.0
MODEL_PATH = Path(__file__).parent.parent / 'models'
path_fooocus_expansion = str(MODEL_PATH / 'fooocus_expansion')

def safe_str(x):
    x = str(x)
    for _ in range(16):
        x = x.replace('  ', ' ')
    return x.strip(",. \r\n")


def remove_pattern(x, pattern):
    for p in pattern:
        x = x.replace(p, '')
    return x


class FooocusExpansion:
    def __init__(self):
        load_file_from_url(
            url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
            model_dir=path_fooocus_expansion,
            file_name='pytorch_model.bin'
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(path_fooocus_expansion)

        positive_words = open(os.path.join(path_fooocus_expansion, 'positive.txt'),
                              encoding='utf-8').read().splitlines()
        positive_words = ['Ġ' + x.lower() for x in positive_words if x != '']

        self.logits_bias = torch.zeros((1, len(self.tokenizer.vocab)), dtype=torch.float32) + neg_inf

        debug_list = []
        for k, v in self.tokenizer.vocab.items():
            if k in positive_words:
                self.logits_bias[0, v] = 0
                debug_list.append(k[1:])

        print(f'Fooocus V2 Expansion: Vocab with {len(debug_list)} words.')

        # debug_list = '\n'.join(sorted(debug_list))
        # print(debug_list)

        # t11 = self.tokenizer(',', return_tensors="np")
        # t198 = self.tokenizer('\n', return_tensors="np")
        # eos = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(path_fooocus_expansion)
        self.model.eval()

        # Device
        device_used = opts.data.get("style_use_gpu",True) and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if device_used else "cpu")
        self.model = self.model.to(self.device)

        use_fp16 = opts.data.get("style_use_fp16",True)
        if use_fp16:
            self.model.half()

        print(f'Fooocus Expansion engine loaded for {self.device}, use_fp16 = {use_fp16}.')

    @torch.no_grad()
    @torch.inference_mode()
    def logits_processor(self, input_ids, scores):
        assert scores.ndim == 2 and scores.shape[0] == 1
        self.logits_bias = self.logits_bias.to(scores)

        bias = self.logits_bias.clone()
        bias[0, input_ids[0].to(bias.device).long()] = neg_inf
        bias[0, 11] = 0

        return scores + bias

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, prompt, seed):
        if prompt == '':
            return ''

        seed = int(seed) % SEED_LIMIT_NUMPY
        set_seed(seed)
        prompt = safe_str(prompt) + ','

        tokenized_kwargs = self.tokenizer(prompt, return_tensors="pt")
        tokenized_kwargs.data['input_ids'] = tokenized_kwargs.data['input_ids'].to(self.device)
        tokenized_kwargs.data['attention_mask'] = tokenized_kwargs.data['attention_mask'].to(self.device)

        current_token_length = int(tokenized_kwargs.data['input_ids'].shape[1])
        max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
        max_new_tokens = max_token_length - current_token_length

        # https://huggingface.co/blog/introducing-csearch
        # https://huggingface.co/docs/transformers/generation_strategies
        features = self.model.generate(**tokenized_kwargs,
                                       top_k=100,
                                       max_new_tokens=max_new_tokens,
                                       do_sample=True,
                                       logits_processor=LogitsProcessorList([self.logits_processor]))

        response = self.tokenizer.batch_decode(features, skip_special_tokens=True)
        result = safe_str(response[0])

        return result
