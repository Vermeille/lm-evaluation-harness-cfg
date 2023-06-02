#!/bin/bash

for cfg in 3; do
    # took Winogrande out, because it's too slow
    rm lm_cache/hf-causal-experimental_pretrained-ComCom-gpt2-medium_use_accelerate-True.db
    CFG=$cfg python3 main.py \
        --model hf-causal-experimental \
        --batch_size auto \
        --model_args pretrained=ComCom/gpt2-medium,use_accelerate=True \
        --tasks hellaswag,winogrande,lambada_openai,boolq,piqa,sciq,arc_easy,arc_challenge,triviaqa \
        --device cuda:0 | tee cfg=$cfg.log
done
