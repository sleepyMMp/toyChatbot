# %%
import glob
import json
from itertools import chain

from datasets import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


# %% config

NUM_PROC = 8
MAX_LENGTH = 1024
BATCH = 10000
tokenizer_path = 'facebook/opt-125m'
exp_dir = 'data/chineseQA'
raw_data_dir = f'{exp_dir}/json'
p0_json_dir = f'{exp_dir}/p0_json'
p1_dataset_dir = f'{exp_dir}/p1_opt_l{MAX_LENGTH}_dataset'

# %% load data

# data_paths = glob.glob(f'{raw_data_dir}/*.json')
# data = list(chain(*[json.load(open(_)) for _ in data_paths]))
ds = load_dataset("hello-simpleai/hc3", data_files=['all.jsonl' ])
ds = ds.select_columns(['index','question', 'human_answers'])



# %% process
# ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
# %% save

ds['train'].to_json(f'{p0_json_dir}/train.json', force_ascii=False)
# ds['test'].to_json(f'{p0_json_dir}/test.json', force_ascii=False)

# %% tokenize

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
eos_token = tokenizer.eos_token

dataset = load_dataset("wangrui6/Zhihu-KOL")
ds = dataset['train'].select(range(10000))
ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)

# %% load data

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
eos_token = tokenizer.eos_token


def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["INSTRUCTION"] = [q.lstrip() for q in examples["INSTRUCTION"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["INSTRUCTION"],
        examples["RESPONSE"],
    )
    # input_ids = tokenized_examples["input_ids"]

    return tokenized_examples


# ds = ds.map(prepare_train_features, batched=True, batch_size=BATCH, num_proc=NUM_PROC, remove_columns=ds['train'].column_names)
ds = ds.map(prepare_train_features, batched=True, remove_columns=ds["train"].column_names)
# ds = ds.shuffle(seed=42)

# example
print(tokenizer.decode(ds['train'][0]['input_ids']))


# save
ds.save_to_disk(p1_dataset_dir)

# python -m exp.poem.process
