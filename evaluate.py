import argparse
import pickle
import time

from moral_summarization.eval import Evaluator


parser = argparse.ArgumentParser()
parser.add_argument('--results-dir',   type=str,  default="results/test_prompts")
parser.add_argument('--summaC',        default=False, action='store_true')
parser.add_argument('--BLANC',         default=False, action='store_true')
parser.add_argument('--QaFactEval',    default=False, action='store_true')
parser.add_argument('--llama',         default=False, action='store_true')
parser.add_argument('--command-r',     default=False, action='store_true')
parser.add_argument('--deepseek',      default=False, action='store_true')
parser.add_argument('--only-test-set', default=False, action='store_true')
parser.add_argument('--seed',          type=str,  default=None)

args = parser.parse_args()

# Make a list of extra metrics to evaluate based on the passed args
extra_metrics = []
if args.summaC:
    extra_metrics.append('summaC')
if args.BLANC:
    extra_metrics.append('BLANC')
if args.QaFactEval:
    extra_metrics.append('QaFactEval')

models = []
if args.llama:
    models.append('Meta-Llama-3-70B-Instruct')
if args.command_r:
    models.append('c4ai-command-r-plus-4bit')
if args.deepseek:
    models.append('DeepSeek-R1-Distill-Qwen-32B')

start_time = time.time()

evaluator = Evaluator(
    models=models,
    extra_metrics=extra_metrics,
    only_test_set=args.only_test_set,
    seed=args.seed
    )

evaluator.evaluate_summaries(args.results_dir)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

with open('dict_of_dfs.pickle', 'wb') as f:
    pickle.dump(evaluator.metrics, f)
