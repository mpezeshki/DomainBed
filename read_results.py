import json
import glob
import argparse
import numpy as np


def read_results_file(file):
    results = {}
    with open(file, "r") as file:
        for line in file:
            res = json.loads(line)
            for key in ['test_envs', 'hparams_seed', 'output_dir', 'dataset', 'algorithm']:
                results[key] = res['args'][key]

            accs = []
            for test_env in res['args']['test_envs']:
                accs += [res[f'env{test_env}_out_acc']]
            te_acc = np.mean(accs)
            results.setdefault('te_acc', []).append(te_acc)
            if 'flip_rate' in res.keys():
                results.setdefault('flip_rate', []).append(res['flip_rate'])

    return results


def best_pt_file(dir, algrthm, dataset, test_envs):

    all_results = {}
    for file in glob.glob(dir + '/*/results.jsonl'):
        results = read_results_file(file)
        if results['dataset'] == dataset and results['algorithm'] == algrthm and results['test_envs'] == test_envs:
            if 'flip_rate' in results.keys():
                # flip_rate at the end of training matters
                all_results.setdefault('flip_rate', []).append(results['flip_rate'][-1])
            # all are the same so just pick the first one
            all_results.setdefault('hparams_seed', []).append(results['hparams_seed'])
            all_results.setdefault('output_dir', []).append(results['output_dir'])
            # choosing the max test_acc
            all_results.setdefault('te_acc', []).append(np.max(results['te_acc']))

    new_results = {}
    for hp in np.unique(all_results['hparams_seed']):
        new_results.setdefault('hparams_seed', []).append(hp)
        new_results.setdefault('flip_rate', []).append(np.mean(np.array(all_results['flip_rate'])[all_results['hparams_seed'] == hp]))
        new_results.setdefault('flip_rate_std', []).append(np.std(np.array(all_results['flip_rate'])[all_results['hparams_seed'] == hp]))

    hp_best = np.array(new_results['hparams_seed'])[np.argmax(new_results['flip_rate'])]
    for hp in np.sort(new_results['hparams_seed']):
        print(hp,
              np.array(new_results['flip_rate'])[np.array(new_results['hparams_seed']) == hp],
              np.array(new_results['flip_rate_std'])[np.array(new_results['hparams_seed']) == hp],
              '(best)' if hp == hp_best else '')
    output_dirs = np.array(all_results['output_dir'])[np.array(all_results['hparams_seed']) == hp_best]
    output_dirs = np.array(all_results['output_dir'])[np.array(all_results['hparams_seed']) == hp_best]
    flip_rates = np.array(all_results['flip_rate'])[np.array(all_results['hparams_seed']) == hp_best]
    print('for all seeds: ')
    for flip_rate, output_dir in zip(flip_rates, output_dirs):
        print(output_dir + '/inferred.pt', flip_rate)
    print('just return the last one: ')
    print(output_dir + '/inferred.pt')
    return output_dir + '/inferred.pt'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read results')
    parser.add_argument('--dir')
    parser.add_argument('--dataset')
    parser.add_argument('--algrthm')
    parser.add_argument('--group_labels', choices=['yes', 'no', 'inferred'])
    parser.add_argument('--selection_criterion', default=None)
    parser.add_argument('--test_envs', action='append')
    args = parser.parse_args()

    best_pt_file(args.dir, args.algrthm, args.dataset, [int(i) for i in args.test_envs])
