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

    def __init__(self, hf_causal_model, instruction_tuned_model=None, cfg=None, round_to=4):
        super().__init__()
        self.hf_causal_model = hf_causal_model
        self.instruction_tuned_model = instruction_tuned_model
        self.cfg = cfg
        self.round_to = round_to

    def arr_to_list(self, torch_arr):
        l = torch_arr.squeeze().cpu().detach().numpy().tolist()
        return list(map(lambda x: round(x, self.round_to), l))

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
        logits_long = self.hf_causal_model(
            cfg_long_seq,
            use_cache=use_cache,
            past_key_values=past_key_values_long
        )
        logits_short = self.hf_causal_model(
            cfg_short_seq,
            use_cache=use_cache,
            past_key_values=past_key_values_short
        )
        if self.instruction_tuned_model is not None:
            # swap devices if necessary
            if self.instruction_tuned_model.device != self.hf_causal_model.device:
                cfg_long_seq = cfg_long_seq.to(self.instruction_tuned_model.device)

            logits_instruct = self.instruction_tuned_model(
                cfg_long_seq,
                use_cache=use_cache,
                past_key_values=past_key_values_instruction_tuned
            )
        else:
            logits_instruct = None

        l = F.log_softmax(logits_long[0][:, -1:], dim=-1)
        s = F.log_softmax(logits_short[0][:, -1:], dim=-1)
        logits_cfg = self.cfg * (l - s) + s

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
                    output_packet['cfg_logits'] = self.arr_to_list(logits_cfg)
                    output_packet['prompted_logits'] = self.arr_to_list(logits_long[0][:, -1:])
                    output_packet['unprompted_logits'] = self.arr_to_list(logits_short[0][:, -1:])
                    if self.instruction_tuned_model is not None:
                        output_packet['instruction_model_logits'] = self.arr_to_list(logits_instruct[0][:, -1:])
                    f.write(json.dumps(output_packet) + '\n')

            # update tokens
            if use_cache:
                running_prompt_tokens = running_unprompted_tokens = continuation_ids[:, i:i+1]
                prompted_kv_cache = logits_long.past_key_values
                unprompted_kv_cache = logits_short.past_key_values
                if logits_instruct is not None:
                    instruct_kv_cache = logits_instruct.past_key_values
            else:
                running_prompt_tokens = torch.cat([running_prompt_tokens, continuation_ids[:, i:i+1]], dim=-1)
                running_unprompted_tokens = torch.cat([running_unprompted_tokens, continuation_ids[:, i:i+1]], dim=-1)

        return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='nomic-ai/gpt4all-j')
    parser.add_argument(
        '--instruction-model', type=str, default='nomic-ai/gpt4all-j',
        help='Secondary model to run against during generation for comparison.'
    )
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--dont-use-instruction', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--device-2', type=str, default=None)
    args = parser.parse_args()
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device_2 is None:
        args.device_2 = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print('loading base model...')
    base_model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision).to(args.device).eval()
    instruction_model = None
    if args.instruction_model is not None:
        print('loading instruction model...')
        instruction_model = AutoModelForCausalLM.from_pretrained(args.instruction_model).to(args.device_2).eval()

    model = CFGModelForCausalLM(
        hf_causal_model=base_model,
        cfg=args.cfg,
        instruction_tuned_model=instruction_model,
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('loading dataset...')
    dataset = pd.read_csv(args.dataset)
    for prompt, continuation in dataset[['inputs_pretokenized', 'targets_pretokenized']].values:
        output_model_name = args.model.replace('/', '-').lower()
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
            f.write(json.dumps({
                'prompt': prompt,
                'model': args.model,
                'instruction-model': args.instruction_model.replace('/', '-').lower(),
            }) + '\n')

        model.gather_logits(
            prompt_ids=prompt_tokens['input_ids'].to(args.device),
            continuation_ids=cont_tokens['input_ids'].to(args.device),
            use_cache=True,
            output_file=output_file,
        )




# python generate_sample.py --cfg 1.5 --instruction-model allenai/tulu-7b --model huggyllama/llama-7b --dont-use-instruction --custom-prompt 0-45
