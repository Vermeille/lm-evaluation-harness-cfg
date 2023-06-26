import matplotlib.pyplot as plt
from gpt2 import models as gpt2_scores
from pythia import models as pythia_scores
from llama import models as llama_scores

models = {**gpt2_scores, **pythia_scores, **llama_scores}

model_to_flops = {
    "GPT2-small": 0.248879616,
    "GPT2-medium": 0.709646336,
    "GPT2-large": 1.54806016,
    "GPT2-xl": 3.1152224,
    "Pythia-70M": 0.089341952,
    "Pythia-160M": 0.247378944,
    "Pythia-410M": 0.70764544,
    "Pythia-1B": 1.81751808,
    "Pythia-1.4B": 2.623250432,
    "Pythia-2.8B": 5.29286144,
    "Pythia-6.9B": 13.301465088,
    "Pythia-12B": 23.17309952,
    "LLaMA-7B": 13.214687232,
    "LLaMA-13B": 25.70404864,
    "LLaMA-30B": 64.631903232,
    "LLaMA-65B": 130.047033344,
}

if __name__ == "__main__":
    tests = list(gpt2_scores['GPT2-small'][1].keys())
    print(tests)
    fig = plt.figure(figsize=(10, 10))
    plt.rc('font', size=14)
    for fignum, test in enumerate(tests):
        plt.subplot(3, 3, fignum + 1)
        #plt.subplots_adjust(hspace=0.1)
        for model, results in models.items():
            # plot accuracies for increasing values of cfg
            tests = list(results[1].keys())
            #tests = [t for t in tests if "ppl" in results[0][t]]
            cfgs = list(results.keys())[1:]
            key = 'acc_norm' if 'acc_norm' in results[1][test] else 'acc'
            key = 'em' if 'em' in results[1][test] else key
            flops = model_to_flops[model]
            #plt.plot([flops, flops * 2], [results[1][test][key]] + [max(results[cfg][test][key] for cfg in cfgs)], 'o-')
            plt.plot([flops], [results[1][test][key]],
                     'o',
                     color='cornflowerblue')
            plt.plot([flops * 2],
                     [max(results[cfg][test][key] for cfg in cfgs)],
                     '*',
                     color='red')
            plt.xlabel("FLOPS (G)")
            plt.xscale('log')
            plt.ylabel(test)
    plt.subplot(3, 3, 1)
    fig.legend(['vanilla', 'CFG'],
               loc='lower right',
               ncol=1,
               bbox_to_anchor=(0.95, 0.05))

    plt.suptitle("Accuracy vs. FLOPS")
    plt.tight_layout()
    plt.savefig("flops.png")
