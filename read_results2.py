import json
import glob
import argparse
import numpy as np
from domainbed import datasets

count = 0
for file in glob.glob('./phase_2_inf/*/results.jsonl'):
    with open(file, "r") as f:
        for line in f:
            res = json.loads(line)
            break
    if res['args']['group_labels'] == 'no':
        count += 1
print(count)
