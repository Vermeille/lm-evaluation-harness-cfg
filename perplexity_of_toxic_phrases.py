from transformers import (AutoModelForCausalLM, AutoTokenizer)
import torch.nn.functional as F
import glob
import torch
from datasets import load_dataset
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from tqdm.auto import tqdm
import json

class CFGModelForCausalLM(nn.Module):
    """Stub of a Model Class that produces the likelihood of a prompt + continuation under a set CFG value."""

    def __init__(self, hf_causal_model, cfg=None):
        super().__init__()
        self.hf_causal_model = hf_causal_model
        self.cfg = cfg

    def forward(self,
                cfg_long_seq,
                cfg_short_seq,
                use_cache=False,
                past_key_values_long=None,
                past_key_values_short=None
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
        l = F.log_softmax(logits_long[0][:, -1:], dim=-1)
        s = F.log_softmax(logits_short[0][:, -1:], dim=-1)
        combined_logits = self.cfg * (l - s) + s

        output = (combined_logits,)
        if use_cache:
            output += (logits_long.past_key_values, logits_short.past_key_values)
        return output

    def calculate_sequence_perplexity(self, prompt_ids, continuation_ids, use_cache=False, *args, **kwargs):
        """Iterates returns the sequence-level perplexity of a `continuation` given a `prompt` using CFG.
        Important: Excludes the first token from the perplexity calculation.
        """

        running_logits = []
        unprompted_kv_cache, prompted_kv_cache = None, None
        running_prompt_tokens = prompt_ids
        running_unprompted_tokens = prompt_ids[:, -1:]
        for i, target_tok in enumerate(continuation_ids.squeeze()):
            logits, prompted_kv_cache, unprompted_kv_cache = self.forward(
                running_prompt_tokens,
                running_unprompted_tokens,
                use_cache=use_cache,
                past_key_values_long=prompted_kv_cache,
                past_key_values_short=unprompted_kv_cache,
            )

            # exclude the first token because it is often skewed due to the way we include the last token from
            # the prompt.
            if i > 0:
                running_logits.append(
                    logits[:, :, target_tok]
                )

            # update tokens
            if use_cache:
                running_prompt_tokens = running_unprompted_tokens = continuation_ids[:, i:i+1]
            else:
                running_prompt_tokens = torch.cat([running_prompt_tokens, continuation_ids[:, i:i+1]], dim=-1)
                running_unprompted_tokens = torch.cat([running_unprompted_tokens, continuation_ids[:, i:i+1]], dim=-1)

        perplexity = -torch.exp(torch.hstack(running_logits).mean())
        return perplexity


class ToxicityDataset(Dataset):
    def __init__(self,
                 toxicity_file='allenai/real-toxicity-prompts',
                 tokenizer=None,
                 toxicity_filter=.5,
                 pos_or_neg='positive',
                 system_prompt=None
                 ):
        real_toxicity_df = (
            load_dataset(toxicity_file)['train']
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
            .loc[lambda df: df['challenging'] == True]
            .loc[:,
                    ['prompt_text',
                     'continuation_text',
                     'prompt_severe_toxicity',
                     ]
            ]
        )
        if pos_or_neg == 'positive':
            self.real_toxicity_df = real_toxicity_df.loc[lambda df: df['prompt_severe_toxicity'] > toxicity_filter]
        else:
            self.real_toxicity_df = real_toxicity_df.loc[lambda df: df['prompt_severe_toxicity'] < toxicity_filter]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.real_toxicity_df)

    def __getitem__(self, idx):
        datapoint = self.real_toxicity_df.iloc[idx]
        prompt_text = datapoint['prompt_text']
        cont_text = datapoint['continuation_text']
        return {
            'prompt_ids': self.tokenizer.encode(prompt_text, return_tensors='pt'),
            'continuation_ids': self.tokenizer.encode(cont_text, return_tensors='pt'),
        }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='allenai/real-toxicity-prompts')
    parser.add_argument('--model', type=str, default="nomic-ai/gpt4all-falcon")
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--system-prompt', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--results-file', type=str, default='results.txt')
    args = parser.parse_args()
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    toxic_continuation_dataset = ToxicityDataset(
        tokenizer=tokenizer,
        toxicity_file=args.dataset,
        system_prompt=args.system_prompt,
        pos_or_neg='positive'
    )
    non_toxic_continuation_dataset = ToxicityDataset(
        tokenizer=tokenizer,
        toxicity_file=args.dataset,
        system_prompt=args.system_prompt,
        toxicity_filter=.2,
        pos_or_neg='negative'
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    model = CFGModelForCausalLM(hf_model).to(args.device)
    cfg_range = [args.cfg]

    for cfg in [1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 7]:
        print('Calculating perplexities...')
        model.cfg = cfg
        for dataset_name, dataset in [
            ('toxic', toxic_continuation_dataset),
            ('non-toxic', non_toxic_continuation_dataset)
        ]:
            all_ppls = []
            for i in tqdm(range(len(dataset))):
                datapoint = dataset[i]
                tensors = {k: v.to(args.device) for k, v in datapoint.items()}
                ppl_for_sequence = model.calculate_sequence_perplexity(**tensors, use_cache=True)
                all_ppls.append(ppl_for_sequence.item())

            all_ppls = list(map(float, all_ppls))
            print(
                f'{dataset_name}. CFG: {cfg}, mean ppl: {sum(all_ppls) / len(all_ppls)}, prompt: {args.system_prompt}'
            )

            with open(f"{args.results_file}", 'a') as f:
                f.write(
                    json.dumps({
                        'cfg': cfg,
                        'mean_ppl': sum(all_ppls) / len(all_ppls),
                        'all_ppls': all_ppls,
                        'prompt': args.system_prompt,
                        'dataset': dataset_name,
                    }) + '\n'
                )
