from transformers import (AutoModelForCausalLM, AutoTokenizer)
from transformers import LogitsWarper, LogitsProcessorList
import torch.nn.functional as F
import glob
from datasets import load_dataset
import pandas as pd
from transformers import (
    LogitsWarper, LogitsProcessorList,
    MinLengthLogitsProcessor, TemperatureLogitsWarper,
    TopKLogitsWarper, TopPLogitsWarper,
    TypicalLogitsWarper,
    RepetitionPenaltyLogitsProcessor
)


def load_toxicity_dataset():
    real_toxicity = load_dataset('allenai/real-toxicity-prompts')
    real_toxicity_df = (
        real_toxicity['train']
             .to_pandas()
             .pipe(lambda df:
                   pd.concat([
                     df['prompt']
                       .pipe(lambda s: pd.DataFrame(s.tolist()))
                       .rename(columns=lambda x: f'prompt_{x}'),
                     df['continuation']
                       .pipe(lambda s: pd.DataFrame(s.tolist()))
                       .rename(columns=lambda x: f'continuation_{x}'),
                    df['challenging']
                   ], axis=1)
                   )
                    [['prompt_text', 'continuation_text', 'prompt_severe_toxicity', 'continuation_severe_toxicity', 'challenging']]
                    .loc[lambda df: df['challenging'] == True]
                    .assign(combined_severe_toxicity=lambda df: df['prompt_severe_toxicity'] * df['continuation_severe_toxicity'])
                    .sort_values('prompt_severe_toxicity', ascending=False)
    )
    return real_toxicity_df


class CFGLogits(LogitsWarper):

    def __init__(self, cfg, inputs_sans_prompt, model):
        self.cfg = cfg
        self.inputs_sans_prompt = inputs_sans_prompt
        self.model = model
        self.out = None

    def __call__(self, input_ids_with_prompt, scores):
        if self.cfg == 1:
            return F.log_softmax(scores, dim=-1)
        scores = F.log_softmax(scores, dim=-1)
        if self.out is None:
            self.out = self.model(self.inputs_sans_prompt, use_cache=True)
        else:
            self.out = self.model(
                input_ids_with_prompt[:, -1:], use_cache=True, past_key_values=self.out.past_key_values
            )
        unconditional_logits = F.log_softmax(self.out.logits[0][-1:], dim=-1)

        out = self.cfg * (scores - unconditional_logits) + unconditional_logits
        return F.log_softmax(out, dim=-1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default="nomic-ai/gpt4all-falcon")
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--remove-bos', action='store_true', help='Test to see whether Llama needs BOS.')
    parser.add_argument('--dont-use-instruction', action='store_true')
    parser.add_argument('--system-prompt', type=str, default=None)
    parser.add_argument('--custom-prompt', nargs='+', default=['0'])
    args = parser.parse_args()

    dataset_df = load_toxicity_dataset()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    for custom_prompt in args.custom_prompt:
        file_pattern = f'assistant-outputs/{args.model.replace("/", "-")}__*.txt'
        num_output_files = len(glob.glob(file_pattern))
        output_fn = f'assistant-outputs/{args.model.replace("/", "-")}__{num_output_files}.txt'

        if custom_prompt.isdigit():
            custom_prompt = dataset_df.iloc[int(custom_prompt)]['prompt_text']

        prompt = (
            f"### Instruction: {args.system_prompt}, "
            f"### Prompt: {custom_prompt}"
            f"### Response:"
        )
        prompt_uncond = (
            f"### Prompt: {custom_prompt}"
            f"### Response:"
        )

        inputs_cfg = tokenizer([prompt], return_tensors="pt")
        inputs_cfg.pop('token_type_ids', None)
        inputs_uncond = tokenizer([prompt_uncond], return_tensors="pt")
        inputs_uncond.pop('token_type_ids', None)
        cfgprocessor = CFGLogits(args.cfg, inputs_uncond['input_ids'], model)

        l = 64
        cfg_outputs = model.generate(
            **inputs_cfg,
            max_new_tokens=l,
            logits_processor=LogitsProcessorList([
                cfgprocessor,
                MinLengthLogitsProcessor(l, eos_token_id=tokenizer.eos_token_id),
                TemperatureLogitsWarper(0.8),
                TopPLogitsWarper(0.95),
            ]),
            do_sample=True,
        )

        prompted_outputs = model.generate(
            **inputs_cfg,
            max_new_tokens=l,
            logits_processor=LogitsProcessorList([
                MinLengthLogitsProcessor(l, eos_token_id=tokenizer.eos_token_id),
                TemperatureLogitsWarper(0.8),
                TopPLogitsWarper(0.95),
            ]),
            do_sample=True,
        )

        with open(output_fn, 'w') as f:
            f.write('### Instruction\n')
            f.write(prompt)
            f.write('\n\n')
            f.write('### CFG\n')
            f.write(tokenizer.decode(cfg_outputs[0]))
            f.write('\n\n')
            f.write('### Prompted\n')
            f.write(tokenizer.decode(prompted_outputs[0]))
            f.write('\n\n')
