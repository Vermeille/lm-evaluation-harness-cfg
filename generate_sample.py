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

    def __init__(self, cfg, inputs, model):
        self.cfg = cfg
        self.inputs = inputs
        self.model = model
        self.out = None

    def __call__(self, input_ids, scores):
        if self.cfg == 1:
            return scores
        prompt_len = len(self.inputs["input_ids"][0])
        scores = F.log_softmax(scores, dim=-1)
        if self.cfg != 1:
            if self.out is None:
                # model p(x_i | x_{<i} ) by cutting out the prompt (except the last letter of it).
                self.out = self.model(input_ids[:, prompt_len - 1:])
            else:
                # model p(x_i | x_{<i} ) by taking the last generation, and then feeding in past_key_values.
                self.out = self.model(input_ids[:, -1:], past_key_values=self.out.past_key_values)

            # get the last layer of logits
            unconditional_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)
            scores = self.cfg * scores + (1 - self.cfg) * unconditional_logits
        return scores

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='nomic-ai/gpt4all-j')
    parser.add_argument('--revision', type=str, default=None)  # "v1.3-groovy"
    parser.add_argument('--remove-bos', action='store_true', help='Test to see whether Llama needs BOS.')
    parser.add_argument('--dont-use-instruction', action='store_true')
    parser.add_argument('--custom-prompt', nargs='+', default=['0'])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision)

    for p in args.custom_prompt:
        user_prompt = p if not p.isdigit() else user_prompts[int(p)]
        prompt_i = len(glob.glob('a-*.txt'))
        if not args.dont_use_instruction:
            prompt = ("### Instruction: The prompt below is a question to answer, "
                      "a task to complete, or a conversation to respond to; decide "
                      "which and write an appropriate response.\n"
                      f"### Prompt: {user_prompt}\n### Response:")
        else:
            prompt = user_prompt

        print(prompt_i, ':', prompt)
        inputs = tokenizer([prompt], return_tensors="pt")

        # processing for Llama
        inputs.pop('token_type_ids', None)
        print('inputs', inputs)
        l = 128

        # with bos
        outputs = model.generate(
            **inputs,
            max_new_tokens=l,
            min_length=l,
            repetition_penalty=1.2,
        )
        with_bos = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # without bos
        inputs['input_ids'] = inputs['input_ids'][:, 1:]
        inputs['attention_mask'] = inputs['attention_mask'][:, 1:]
        outputs = model.generate(
            **inputs,
            max_new_tokens=l,
            min_length=l,
            repetition_penalty=1.2,
            # logits_processor=LogitsProcessorList([CFGLogits(1.5, inputs, model)]),
        )
        without_bos = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # shuf = ['cfg', 'no_cfg']
        # random.shuffle(shuf)

        with open(f'a-{prompt_i}-with-bos.txt', 'w') as f:
            print(prompt, file=f)
            print(with_bos, file=f)

        with open(f'a-{prompt_i}-without-bos.txt', 'w') as f:
            print(prompt, file=f)
            print(without_bos, file=f)

