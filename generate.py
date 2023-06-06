import sys
from transformers import (GPT2Tokenizer, AutoModelForCausalLM,
                          GPTNeoXForCausalLM, AutoTokenizer)
import numpy as np
import torch
from transformers import (LogitsWarper, LogitsProcessorList,
                          MinLengthLogitsProcessor, TemperatureLogitsWarper,
                          TopKLogitsWarper, TopPLogitsWarper,
                          TypicalLogitsWarper,
                          RepetitionPenaltyLogitsProcessor)
from transformers.generation import LogitNormalization
import torch.nn.functional as F


class FixedRepetitionPenaltyLogitsProcessor(RepetitionPenaltyLogitsProcessor):

    def __call__(self, input_ids, scores):
        scores -= scores.min()
        return super().__call__(input_ids, scores)


def entropy(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * logits).sum(-1)


class CFGLogits(LogitsWarper):

    def __init__(self, cfg, inputs, model):
        self.cfg = cfg
        self.inputs = inputs
        self.model = model
        self.out = None

    def __call__(self, input_ids, scores):
        if self.cfg == 1:
            return F.log_softmax(scores, dim=-1)
        scores = F.log_softmax(scores, dim=-1)
        if self.out is None:
            self.out = self.model(self.inputs, use_cache=True)
        else:
            self.out = self.model(input_ids[:, -1:],
                                  use_cache=True,
                                  past_key_values=self.out.past_key_values)
        unconditional_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)
        out = self.cfg * (scores - unconditional_logits) + unconditional_logits
        return F.log_softmax(out, dim=-1)


class CFGLogitsNGram(LogitsWarper):

    def __init__(self, cfg, model, ngram):
        self.cfg = cfg
        self.model = model
        self.ngram = ngram

    def __call__(self, input_ids, scores):
        if self.cfg == 1:
            return F.log_softmax(scores, dim=-1)
        scores = F.log_softmax(scores, dim=-1)
        out = self.model(input_ids[:, -self.ngram:])
        unconditional_logits = F.log_softmax(out.logits[0][-1:], dim=-1)
        out = self.cfg * (scores - unconditional_logits) + unconditional_logits
        return F.log_softmax(out, dim=-1)


select = sys.argv[1]
if select == 'gpt2-large' or select == 'gpt2-medium':
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(select)
    model.eval()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    prompt = sys.argv[2]
    inputs_normal = tokenizer([prompt], return_tensors="pt")
    inputs_cfg = inputs_normal['input_ids'][:, -1:]
elif select == 'pythia':
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-2.8b-deduped", )

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m-deduped", )
    prompt = sys.argv[2]
    inputs_normal = tokenizer([prompt], return_tensors="pt")
    inputs_cfg = inputs_normal['input_ids'][:, -1:]
elif select == 'gpt4all':
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/gpt4all-j")
    model = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j",
                                                 revision="v1.2-jazzy")
    prompt_normal = (
        "### Instruction: The prompt below is a question to answer, "
        "a task to complete, or a conversation to respond to; decide "
        "which and write an appropriate response.\n"
        f"### Prompt: {sys.argv[2]}\n### Response:")
    prompt_cfg = (
        "### Instruction: The prompt below is a question to answer, "
        "a task to complete, or a conversation to respond to; decide "
        "which and write an appropriate response.\n"
        f"### Prompt:\n### Response:")
    prompt_cfg = ':'

    inputs_normal = tokenizer([prompt_normal], return_tensors="pt")
    inputs_cfg = tokenizer([prompt_cfg], return_tensors="pt")['input_ids']
else:
    raise ValueError('unknown model')


def gen(cfg=1, ngram=0):
    l = 256
    torch.manual_seed(52846274852342)
    if ngram == 0:
        cfgprocessor = CFGLogits(cfg, inputs_cfg, model)
    else:
        cfgprocessor = CFGLogitsNGram(cfg, model, ngram)
    outputs = model.generate(
        **inputs_normal,
        max_new_tokens=l,
        logits_processor=LogitsProcessorList([
            cfgprocessor,
            MinLengthLogitsProcessor(l, eos_token_id=tokenizer.eos_token_id),
            TemperatureLogitsWarper(0.8),
            TopPLogitsWarper(0.95),
        ]),
        do_sample=True,
    )
    print('cfg', cfg, 'ngram', ngram if ngram > 0 else 'prompt')
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    print('=======')


gen(1)
gen(1.5)
gen(1.2, ngram=1)
gen(1.2, ngram=2)
gen(1.2, ngram=3)
