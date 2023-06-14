import os.path
import random
import sys
from transformers import (GPT2Tokenizer, AutoModelForCausalLM,
                          GPTNeoXForCausalLM, AutoTokenizer)
import numpy as np
import torch
import pandas as pd
from transformers import LogitsWarper, LogitsProcessorList
from transformers.generation import LogitNormalization
import torch.nn.functional as F
import glob
import json
from torch import nn


class CFGModelForCausalLM(nn.Module):
    """Stub of a Model Class that produces the likelihood of a prompt + continuation under a set CFG value."""

    def __init__(self, hf_causal_model=None, model_family=None, instruction_tuned_model=None, cfg=None, round_to=4):
        super().__init__()
        self.hf_causal_model = hf_causal_model
        self.instruction_tuned_model = instruction_tuned_model
        self.cfg = cfg
        self.round_to = round_to
        self.model_family = model_family

    def arr_to_list(self, torch_arr):
        l = torch_arr.squeeze().cpu().detach().numpy().tolist()
        return list(map(lambda x: round(x, self.round_to), l))

    def model_forward(self, input_ids, model, use_cache=False, past_key_values=None):
        if self.model_family == 't5':
            return model(input_ids=input_ids, decoder_input_ids=input_ids, use_cache=use_cache, past_key_values=past_key_values)
        else:
            return model(input_ids=input_ids, use_cache=use_cache, past_key_values=past_key_values)

    def forward(self,
                cfg_long_seq,
                cfg_short_seq,
                use_cache=False,
                past_key_values_long=None,
                past_key_values_short=None,
                past_key_values_instruction_tuned=None,
    ):
        """Generic `forward` method for calculating the logits of a sequence using CFG sequence.
        Left general so that
        """
        logits_cfg = logits_long = logits_short = None
        if self.hf_causal_model is not None:
            logits_long = self.model_forward(
                model=self.hf_causal_model,
                input_ids=cfg_long_seq,
                use_cache=use_cache,
                past_key_values=past_key_values_long
            )
            logits_short = self.model_forward(
                model=self.hf_causal_model,
                input_ids=cfg_short_seq,
                use_cache=use_cache,
                past_key_values=past_key_values_short
            )

            l = F.log_softmax(logits_long[0][:, -1:], dim=-1)
            s = F.log_softmax(logits_short[0][:, -1:], dim=-1)
            logits_cfg = self.cfg * (l - s) + s

        logits_instruct = None
        if self.instruction_tuned_model is not None:
            # swap devices if necessary
            if self.instruction_tuned_model.device != self.hf_causal_model.device:
                cfg_long_seq = cfg_long_seq.to(self.instruction_tuned_model.device)

            logits_instruct = self.model_forward(
                model=self.instruction_tuned_model,
                input_ids=cfg_long_seq,
                use_cache=use_cache,
                past_key_values=past_key_values_instruction_tuned
            )

        return (
            logits_cfg, logits_long, logits_short, logits_instruct
        )

    def gather_logits(self, prompt_ids, continuation_ids, use_cache=False, output_file=None, *args, **kwargs):
        """Iterates returns the sequence-level perplexity of a `continuation` given a `prompt` using CFG.
        Important: Excludes the first token from the perplexity calculation.
        """
        unprompted_kv_cache, prompted_kv_cache, instruct_kv_cache = None, None, None
        running_prompt_tokens = prompt_ids
        running_unprompted_tokens = prompt_ids[:, -1:]
        for i, target_tok in enumerate(continuation_ids.squeeze()):
            logits_cfg, logits_long, logits_short, logits_instruct = self.forward(
                running_prompt_tokens,
                running_unprompted_tokens,
                use_cache=use_cache,
                past_key_values_long=prompted_kv_cache,
                past_key_values_short=unprompted_kv_cache,
                past_key_values_instruction_tuned=instruct_kv_cache,
            )

            if output_file is not None:
                with open(output_file, 'a') as f:
                    output_packet = {}
                    if self.hf_causal_model is not None:
                        output_packet['cfg_logits'] = self.arr_to_list(logits_cfg)
                        output_packet['prompted_logits'] = self.arr_to_list(logits_long[0][:, -1:])
                        output_packet['unprompted_logits'] = self.arr_to_list(logits_short[0][:, -1:])
                    if self.instruction_tuned_model is not None:
                        output_packet['instruction_model_logits'] = self.arr_to_list(logits_instruct[0][:, -1:])
                    f.write(json.dumps(output_packet) + '\n')

            # update tokens
            if use_cache:
                running_prompt_tokens = running_unprompted_tokens = continuation_ids[:, i:i+1]
                if logits_long is not None:
                    prompted_kv_cache = logits_long.past_key_values
                    unprompted_kv_cache = logits_short.past_key_values
                if logits_instruct is not None:
                    instruct_kv_cache = logits_instruct.past_key_values
            else:
                running_prompt_tokens = torch.cat([running_prompt_tokens, continuation_ids[:, i:i+1]], dim=-1)
                running_unprompted_tokens = torch.cat([running_unprompted_tokens, continuation_ids[:, i:i+1]], dim=-1)

        return None


def load_model(model_name, revision, device):
    if ('t5' in model_name) or ('T0' in model_name):
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).to(device).eval()
    return tokenizer, model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument(
        '--instruction-model', type=str, default=None,
        help='Secondary model to run against during generation for comparison.'
    )
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--dont-use-instruction', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--device-2', type=str, default=None)
    parser.add_argument('--max-prompt-len', type=int, default=100)
    args = parser.parse_args()
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device_2 is None:
        args.device_2 = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model is not None:
        output_model_name = args.model.replace('/', '-').lower()
    else:
        output_model_name = args.instruction_model.replace('/', '-').lower()

    print('loading base model...')
    base_model = None
    if args.model is not None:
        tokenizer, base_model = load_model(args.model, args.revision, args.device)

    instruction_model = None
    if args.instruction_model is not None:
        print('loading instruction model...')
        tokenizer, instruction_model = load_model(args.instruction_model, args.revision, args.device_2)

    model = CFGModelForCausalLM(
        hf_causal_model=base_model,
        cfg=args.cfg,
        instruction_tuned_model=instruction_model,
        model_family='t5' if 't5' in args.model else 'decoder',
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('loading dataset...')
    dataset = pd.read_csv(args.dataset)
    for prompt, continuation in dataset[['inputs_pretokenized', 'targets_pretokenized']].values:
        prompt_i = len(glob.glob(f'{args.output_dir}/logit-files__{output_model_name}__*.txt'))
        output_file = f'{args.output_dir}/logit-files__{output_model_name}__{prompt_i}.txt'

        if not args.dont_use_instruction:
            prompt = ("### Instruction: The prompt below is a question to answer, "
                      "a task to complete, or a conversation to respond to; decide "
                      "which and write an appropriate response.\n"
                      f"### Prompt: {prompt}\n### Response:")

        print(prompt_i, ':', prompt)
        prompt_tokens = tokenizer([prompt], return_tensors="pt")
        cont_tokens = tokenizer([continuation], return_tensors="pt")
        print('inputs', prompt_tokens)
        with open(output_file, 'a') as f:
            output_header = {}
            output_header['prompt'] = prompt
            if args.model is not None:
                output_header['model'] = args.model.replace('/', '-').lower()
            if args.instruction_model is not None:
                output_header['instruction-model'] = args.instruction_model.replace('/', '-').lower()
            f.write(json.dumps(output_header) + '\n')

        model.gather_logits(
            prompt_ids=prompt_tokens['input_ids'].to(args.device),
            continuation_ids=cont_tokens['input_ids'].to(args.device),
            use_cache=True,
            output_file=output_file,
        )




# python log_logits_on_p3.py --cfg 1.5 --instruction-model  allenai/tulu-7b --model huggyllama/llama-7b --dont-use-instruction --dataset dataset_to_generate_on.csv --output-dir p3-output-dir/ --device cuda:2 --device-2 cuda:3
# python log_logits_on_p3.py --cfg 1.5 --instruction-model bigscience/T0pp --model t5-11b --dont-use-instruction --dataset dataset_to_generate_on.csv --output-dir p3-output-dir/ --device cuda:4 --device-2 cuda:5
