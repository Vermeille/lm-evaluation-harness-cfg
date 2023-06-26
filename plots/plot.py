import matplotlib.pyplot as plt


def table(models):
    for model, results in models.items():
        # plot accuracies for increasing values of cfg
        longest_test = max([len(t) for t in results[1].keys()])

        def pad(s):
            return s + " " * (longest_test - len(s))

        tests = list(results[1].keys())
        #tests.remove('triviaqa')
        #tests = [t for t in tests if "ppl" in results[0][t]]
        cfgs = list(results.keys())[0:]
        print(model)
        print('===')
        print(pad('CFG'), '\t'.join(str(c) for c in cfgs))
        for test in tests:
            key = 'acc_norm' if 'acc_norm' in results[1][test] else 'acc'
            print(
                pad(test), '\t'.join([
                    f'{100 * results[cfg][test][key]:>5.2f}' for cfg in cfgs
                ]))
        print()


def latex_table(models):
    for tests in [['arc_challenge', 'arc_easy', 'boolq', 'hellaswag'],
                  ['piqa', 'sciq', 'winogrande', 'lambada_openai']]:
        print(" & ".join(tests))
        for model, results in models.items():
            # plot accuracies for increasing values of cfg
            longest_test = max([len(t) for t in results[1].keys()])

            def pad(s):
                return s + " " * (longest_test - len(s))

            #tests = [t for t in tests if "ppl" in results[0][t]]
            cfgs = list(results.keys())[0:]
            print(model, end='')
            for test in tests:
                key = 'acc_norm' if 'acc_norm' in results[1][test] else 'acc'
                key = "em" if "em" in results[1][test] else key
                base = results[1][test][key]
                #best = max([results[cfg][test][key] for cfg in cfgs[1:]])
                best = results[1.5][test][key]
                if best > base:
                    pre_best = '\\textbf{'
                    post_best = '}'
                    pre_base = ''
                    post_base = ''
                else:
                    pre_best = ''
                    post_best = ''
                    pre_base = '\\textbf{'
                    post_base = '}'
                print(' & ', f'{pre_base}{100 * base:>5.1f}{post_base} /'
                      f' {pre_best}{100 * best:>5.1f}{post_best}',
                      end='')
            print('\\\\')


def plot(models, name):
    tests = list(next(iter(models.values()))[1].keys())
    print(name, tests)
    #tests.remove('triviaqa')
    fig = plt.figure(figsize=(10, 10))
    plt.rc('font', size=14)
    for fignum, test in enumerate(tests):
        plt.subplot(3, 3, fignum + 1)
        #plt.subplots_adjust(hspace=0.1)
        for model, results in models.items():
            tests = list(results[1].keys())
            cfgs = list(results.keys())[0:]
            key = 'acc_norm' if 'acc_norm' in results[1][test] else 'acc'
            key = "em" if "em" in results[1][test] else key
            plt.plot(cfgs, [results[cfg][test][key] for cfg in cfgs], 'o-')
            plt.ylabel(test)
    fig.legend(list(models.keys()),
               loc='lower right',
               ncol=1,
               bbox_to_anchor=(0.95, 0.05))
    plt.suptitle(f"Results for {name}")
    plt.tight_layout()
    plt.savefig(f"{name.lower()}-split.png")


if __name__ == '__main__':
    from llama import models as llama_models
    from pythia import models as pythia_models
    from gpt2 import models as gpt2_models
    plot(llama_models, 'LLaMA')
    plot(pythia_models, 'Pythia')
    plot(gpt2_models, 'GPT2')
    latex_table(llama_models)
    latex_table(pythia_models)
    latex_table(gpt2_models)
    #table(gpt2_models)
