import random
import sys
from transformers import (GPT2Tokenizer, AutoModelForCausalLM,
                          GPTNeoXForCausalLM, AutoTokenizer)
import numpy as np
import torch
from transformers import LogitsWarper, LogitsProcessorList
from transformers.generation import LogitNormalization
import torch.nn.functional as F
import glob
import json

user_prompts = [
    'Why did the chicken cross the road?',
    'What is the meaning of life?',
    'What is the answer to life, the universe, and everything?',
    'What is the best way to cook a steak?',
    'How do you make a pizza?',
    'What is the best way to make a pizza?',
    'Why is the sky blue?',
    'Who is the best basketball player of all time?',
    'What are trans fats?',
    'What are transformers?',
    'What are neural networks?',
    'What is the best way to learn a language?',
    'Who is Optimus Prime?',
    'Write a haiku about the meaning of life.',
    'Write the python code to print the first 100 prime numbers.',
    'Give me a recipe for a delicious meal.',
    'How to implement authentication with Flask?',
    'What is the easiest python library to bootstrap a web app?',
    'I am in France and I want to be polite, give me some advice.',
    'Is Yann LeCun the father of deep learning?',
    'Is Yann LeCun the father of convolutional neural networks?',
    'Is Yann LeCun great because he is French, or is he French because he is great?',
    'Is Yann LeCun great because he is French, or despite being French?',
    'Explain the algorithm AlphaZero in few sentences.',
    'I want to learn how to play chess, what is the best way to start?',
    'How are metal vocalists able to scream for so long?',
    'What is the best way to learn how to sing?',
    'What is the best way to learn how to play the guitar?',
    'Give me compelling ideas for a startup.',
    'Give me compelling ideas for a D&D campaign in a medfan version of Italy.',
    'Give me compelling ideas for a D&D campaign in a medfan version of Greece.',
    'Give me compelling ideas for a D&D campaign in a medfan version of France.',
    'Write the lyrics of a death metal song about chickens.',
    'Write the lyrics of a death metal song about AI research.',
    'What kind of present should I buy for my 30yo wife who loves dancing, D&D, board games, and soft metal music?',
    'What kind of present should I buy for my 30yo husband who loves AI, D&D, board games, and metal music?',
    'Are nerds trendy?',
    'What is a taxonomy?',
    'What are the main differences between driving in France and in the US?',
    'Who are artists that are similar to Gojira?',
    'Who are artists that are famous in the US but not abroad?',
    'Suggest a unique and compelling plot for a scifi novel where people can text each other through time.',
    'Suggest a unique and compelling plot for a scifi novel where people can text each other through time, but only in the past.',
    'What was the Cambridge Analytica scandal?',
    'How to choose a good learning rate?',
]


class CFGLogits(LogitsWarper):

    def __init__(self, cfg, inputs, model, logits_output_file=None, second_model=None):
        self.cfg = cfg
        self.inputs = inputs
        self.model = model
        self.second_model = second_model  # to optionally compare the logits we would have gotten with an
                                          # instruction-tuned model
        self.previous_output = None
        self.previous_output_second = None
        self.logits_output_file = logits_output_file

    def arr_to_list(self, torch_arr):
        return torch_arr.cpu().numpy()[0].tolist()

    def run_model(self, input_ids, prompt_len, previous_output, model):
        if previous_output is None:
            # model p(x_i | x_{<i} ) by cutting out the prompt (except the last letter of it).
            previous_output = model(input_ids[:, prompt_len - 1:])
        else:
            # model p(x_i | x_{<i} ) by taking the last generation, and then feeding in past_key_values.
            previous_output = model(input_ids[:, -1:], past_key_values=previous_output.past_key_values)

        return previous_output

    def __call__(self, input_ids, prompted_logits):
        if self.cfg == 1:
            return prompted_logits

        prompt_len = len(self.inputs["input_ids"][0])
        prompted_logits = F.log_softmax(prompted_logits, dim=-1)
        self.previous_output = self.run_model(input_ids, prompt_len, self.previous_output, self.model)

        # get the last layer of logits
        unconditional_logits = F.log_softmax(self.previous_output.logits[0][-1:], dim=-1)
        cfg_logits = self.cfg * prompted_logits + (1 - self.cfg) * unconditional_logits
        if self.second_model is not None:
            self.previous_output_second = self.run_model(input_ids, prompt_len, self.previous_output_second, self.second_model)
            instruction_logits = F.log_softmax(self.previous_output_second.logits[0][:prompt_len], dim=-1)

        if self.logits_output_file is not None:
            with open(self.logits_output_file, 'a') as f:
                output_packet = {}
                output_packet['cfg_logits'] = self.arr_to_list(cfg_logits)
                output_packet['prompted_logits'] = self.arr_to_list(prompted_logits)
                output_packet['unprompted_logits'] = self.arr_to_list(unconditional_logits)
                if self.second_model is not None:
                    output_packet['second_model_logits'] = self.arr_to_list(instruction_logits)
                f.write(json.dumps(output_packet) + '\n')

        return cfg_logits


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
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--remove-bos', action='store_true', help='Test to see whether Llama needs BOS.')
    parser.add_argument('--dont-use-instruction', action='store_true')
    parser.add_argument('--system-prompt', type=str, default=None)
    parser.add_argument('--custom-prompt', nargs='+', default=['0'])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision)

    for p in args.custom_prompt:
        output_model_name = args.model.replace('/', '-')
        user_prompt = p if not p.isdigit() else user_prompts[int(p)]
        prompt_i = len(glob.glob(f'logit-files__{output_model_name}__*.txt'))
        output_file = f'logit-files__{output_model_name}__{prompt_i}.txt'
        if not args.dont_use_instruction:
            prompt = ("### Instruction: The prompt below is a question to answer, "
                      "a task to complete, or a conversation to respond to; decide "
                      "which and write an appropriate response.\n"
                      f"### Prompt: {user_prompt}\n### Response:")
        else:
            prompt = user_prompt

        print(prompt_i, ':', prompt)
        inputs = tokenizer([prompt], return_tensors="pt")

        with open(output_file, 'w') as f:
            print(prompt, file=f)

        # processing for Llama
        inputs.pop('token_type_ids', None)
        print('inputs', inputs)
        l = 128

        outputs = model.generate(
            **inputs,
            max_new_tokens=l,
            min_length=l,
            repetition_penalty=1.2,
            logits_processor=LogitsProcessorList([CFGLogits(1.5, inputs, model, logits_output_file=output_file)]),
        )