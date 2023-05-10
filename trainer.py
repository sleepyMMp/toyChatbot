import os
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from os import path as osp
from pprint import pprint

from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer
from transformers import HfArgumentParser
from transformers import TrainingArguments

sys.path.append(osp.join(osp.dirname(__file__), '..'))

# %% config

@dataclass
class Args:
    data_path: str = field()
    model_path: str = field()
    tokenizer_path: str = ''

    model_class_name: str = 'auto'
    max_length: int = 1024

    output_dir: str = ''
    num_train_epochs: int = 1
    total_batch_size: int = -1
    mini_batch_size: int = -1
    gradient_checkpointing: bool = False

    optim: str = 'adamw_torch_fused'
    fp16_opt_level: str = 'O1'  # ['O0', 'O1', 'O2', and 'O3']

    project_name: str = ''
    run_name: str = ''

    learning_rate: float = 3e-5

    torch_compile: bool = True

    def __post_init__(self):
        if not self.tokenizer_path:
            self.tokenizer_path = self.model_path

        if self.mini_batch_size == -1:
            self.total_batch_size = self.mini_batch_size
        self.gradient_accumulation_steps = self.total_batch_size // self.mini_batch_size

        self.learning_rate = float(self.learning_rate)

        # wandb
        os.environ['WANDB_PROJECT'] = self.project_name
        self.run_name += f'-{time.time()}'

parser = ArgumentParser()
parser.add_argument('--config_path', '-c', type=str)
cli_args = parser.parse_args()

hf_parser = HfArgumentParser([Args])
args: Args = hf_parser.parse_yaml_file(cli_args.config_path)[0]

pprint(args.__dict__)

# %%
# train_inputs, train_targets = dataset.dataset()
from datasets import load_dataset
data = load_dataset("opus100", "en-zh")


# %% model
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
tokenizer.max_length = args.max_length

if args.model_class_name == 'auto':
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
elif args.model_class_name == 'flash opt':
    from models import OPTForCausalLM

    model = OPTForCausalLM.from_pretrained(args.model_path)
else:
    raise ValueError(f'Unknown model class name: {args.model_class_name}')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# %% trainer
train_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=False,

    # data
    per_device_train_batch_size=args.mini_batch_size,
    per_device_eval_batch_size=args.mini_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    dataloader_num_workers=2,
    gradient_checkpointing=args.gradient_checkpointing,

    # precision
    bf16=True,
    bf16_full_eval=True,
    tf32=True,
    fp16_opt_level=args.fp16_opt_level,

    # eval
    evaluation_strategy='steps',
    save_strategy='steps',
    eval_steps=100,
    save_steps=100,
    save_total_limit=1,
    metric_for_best_model='eval_loss',
    prediction_loss_only=True,

    # log
    logging_strategy='steps',
    logging_steps=10,
    logging_dir=args.output_dir,
    logging_first_step=True,
    run_name=args.run_name,
    report_to=['wandb'],

    # hyper-parameters
    learning_rate=args.learning_rate,
    lr_scheduler_type='constant',
    num_train_epochs=args.num_train_epochs,
    warmup_steps=0,
    weight_decay=0.0,
    optim=args.optim,

    greater_is_better=False,

    # flags
    do_train=True,
    do_eval=True,
    do_predict=False,
    load_best_model_at_end=True,
    torch_compile=args.torch_compile,
    seed=42,
)

trainer = Trainer(
    args=train_args,
    model=model,
    data_collator=data_collator,
    train_dataset=data['train'],
    eval_dataset=data['test'],

)


trainer.train()
trainer.save_model(output_dir=f'{args.output_dir}/best')
tokenizer.save_pretrained(f'{args.output_dir}/best')