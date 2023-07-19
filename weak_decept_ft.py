from argparse import ArgumentParser
from datasets import DatasetDict, load_from_disk
from itertools import islice
from peft import get_peft_model, LoraConfig, TaskType, PeftType
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
import torch
import wandb


wandb.login()


parser = ArgumentParser()
parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-160m")
parser.add_argument("--ds-name", type=str, default="./custom-datasets/AkariAsai/PopQA_erroneous_multi_template")
parser.add_argument("--max-length", type=int, default=1024)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--n-epochs", type=int, default=20)
parser.add_argument("--warmup-steps", type=int, default=400)
parser.add_argument("--eval-interval", type=int, default=200, help="measure val set every n batches")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--n-train", type=int, default=-1)
parser.add_argument("--n-val", type=int, default=-1)
parser.add_argument("--n-test", type=int, default=-1)
parser.add_argument("--lora-rank", type=int, default=32)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--lora-dropout", type=float, default=0.1)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--no-peft", action="store_true")
parser.add_argument("--target-modules", nargs="+", default=["dense_h_to_4h", "dense_4h_to_h", "query_key_value"])
parser.add_argument("--template", type=str, default="{}")
parser.add_argument("--verbalizers", nargs="+", default=["no", "yes"], help="verbalizers for each label. They should be one token each.")

args = parser.parse_args()

model_name = args.model_name
ds_name = args.ds_name

max_length = args.max_length
lr = args.lr
num_epochs = args.n_epochs
warmup_steps = args.warmup_steps
eval_interval = args.eval_interval
batch_size = args.batch_size
weight_decay = args.weight_decay
n_train = args.n_train
n_val = args.n_val
n_test = args.n_test
lora_rank = args.lora_rank
lora_alpha = args.lora_alpha
lora_dropout = args.lora_dropout
device = args.device
use_peft = not args.no_peft
target_modules = args.target_modules
template = args.template
verbalizers = args.verbalizers  # need to be one token


# config for wandb
cfg = vars(args)
cfg["verbalizers"] = verbalizers
cfg["template"] = template


### LOAD/PROCESS DATASET, AND TRAIN MODEL ###

# load dataset
ds = load_from_disk(ds_name)
ds

ds["train"] = ds["train"].shuffle()
ds["test"] = ds["test"].shuffle()

n_train = len(ds["train"]) if n_train == -1 else n_train
n_val = len(ds["validation"]) if n_val == -1 else n_val
n_test = len(ds["test"]) if n_test == -1 else n_test
ds = DatasetDict({
    "train": ds["train"].select(range(n_train)),
    "validation": ds["validation"].select(range(n_val)),
    "test": ds["test"].select(range(n_test))
})

# instantiate tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# define templatize and tokenize functions
def tokenize_examples(examples):
    batch_size = len(examples["text"])
    print(batch_size)

    # apply template to each example
    texts = [template.format(text) for text in examples["text"]]
    targets = [verbalizers[label] for label in examples["label"]]
    
    # tokenize inputs and targets
    inputs = tokenizer(texts)
    labels = tokenizer(targets)

    # concatenate inputs and labels
    for i in range(batch_size):
        sample_input_ids = inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        # be careful that the correct whitespace is between the two parts
        inputs["input_ids"][i] = sample_input_ids + label_input_ids
        # when a label is -100, the corresponding loss is ignored
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        # 1 means attend to the token
        inputs["attention_mask"][i] = [1] * len(inputs["input_ids"][i])
    print(max([len(input_ids) for input_ids in inputs["input_ids"]]))

    # pad everything to max_length and convert to tensors
    for i in range(batch_size):
        sample_input_ids = inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        inputs["input_ids"][i] = torch.tensor(inputs["input_ids"][i][:max_length])
        inputs["attention_mask"][i] = torch.tensor(inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        
    inputs["labels"] = labels["input_ids"]
    print(tokenizer.decode(inputs["input_ids"][0]))
    return inputs


def tokenize_eval_examples(examples):
    # similar to tokenize_examples, but without the label

    batch_size = len(examples["text"])

    # apply template to each example
    inputs = [template.format(text) for text in examples["text"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs)
    
    # pad everything to max_length and convert to tensors
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
    
    out_dict = model_inputs
    out_dict["labels"] = torch.tensor(examples["label"])
    out_dict["true_labels"] = torch.tensor(examples["true_label"])
    return out_dict


# templateize and tokenize train
train_encodings = ds["train"].map(
    tokenize_examples,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)
train_eval_encodings = ds["train"].select(range(n_val)).map(
    tokenize_eval_examples,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = train_encodings
train_eval_dataset = train_eval_encodings

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
train_eval_dataloader = DataLoader(
    train_eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

# validation and test
eval_encodings = ds.map(
    tokenize_eval_examples,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

eval_dataset = eval_encodings["validation"]
test_dataset = eval_encodings["test"]

eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
if use_peft:
    peft_config = LoraConfig(
        peft_type=PeftType.LORA, task_type=TaskType.CAUSAL_LM,
        inference_mode=False, target_modules=target_modules,
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
model = model.to(device)  # we want to keep the lora params in single precision, so don't call half() after pefting

num_erroneous = 0
for row in ds["validation"]:
    if row["label"] != row["true_label"]:
        num_erroneous += 1

print(f"Number of erroneous examples in val: {num_erroneous} ({num_erroneous / len(ds['validation']) * 100:.2f}%)")

wandb.init(
    project="weak-deception",
    name=f"{'LoRA' if use_peft else 'ft'}-{model_name}-{ds_name}",
        
    # track hyperparameters and run metadata
    config=cfg
)

def logits_to_text(logits):
    tok_false, tok_true = tokenizer(verbalizers[0])["input_ids"], tokenizer(verbalizers[1])["input_ids"]
    assert len(tok_false) == len(tok_true) == 1
    tok_false, tok_true = tok_false[0], tok_true[0]
    p_false, p_true = logits[:, -1, [tok_false, tok_true]].softmax(dim=-1).unbind(dim=-1)
    return [verbalizers[0] if p_false > p_true else verbalizers[1] for p_false, p_true in zip(p_false, p_true)]


def ids_to_text(ids):
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


def eval_model(use_tqdm=False, dataloader=eval_dataloader):
    model.eval()
    preds = []
    labels = []
    is_erroneous = []

    iterator = tqdm(dataloader) if use_tqdm else dataloader
    for batch in iterator:
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits
            text_preds = logits_to_text(logits)

            ps = [p == verbalizers[1] for p in text_preds]
            labs = batch["labels"].tolist()
            true_labs = batch["true_labels"].tolist()
            is_err = [labs[i] != true_labs[i] for i in range(len(labs))]

            preds.extend(ps)
            labels.extend(labs)
            is_erroneous.extend(is_err)
    
    preds, labels, is_erroneous = np.array(preds), np.array(labels), np.array(is_erroneous)
    acc = accuracy_score(labels, preds)
    acc_err = accuracy_score(labels[is_erroneous], preds[is_erroneous])
    acc_non_err = accuracy_score(labels[~is_erroneous], preds[~is_erroneous])
            
    return acc, acc_err, acc_non_err

acc, acc_err, acc_non_err = eval_model(use_tqdm=False)
print(f"Initial Acc: {acc}, Acc on erroneous: {acc_err}, Acc on non-erroneous: {acc_non_err}")

# only the LORA parameters should be updated
learnable_parameters = [p for p in model.parameters() if p.requires_grad]
print(f"Number of learnable parameters: {len(learnable_parameters)}")
optimizer = AdamW(learnable_parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))  # adam beta2 default is 0.999

lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if (step + 1) % eval_interval == 0:
            acc, acc_err, acc_non_err = eval_model(use_tqdm=False)
            print(f"Acc: {acc}, Acc on erroneous: {acc_err}, Acc on non-erroneous: {acc_non_err}")

            train_acc, train_acc_err, train_acc_non_err = eval_model(use_tqdm=False, dataloader=train_eval_dataloader)
            print(f"Train Acc: {train_acc}, Train Acc on erroneous: {train_acc_err}, Train Acc on non-erroneous: {train_acc_non_err}")
            wandb.log({"train_acc": train_acc, "train_acc_err": train_acc_err, "train_acc_non_err": train_acc_non_err, "train_loss": total_loss / step, "step": step, "epoch": epoch, "acc": acc, "acc_err": acc_err, "acc_non_err": acc_non_err})

            model.train()
            
    print("Epoch {} loss: {}".format(epoch, total_loss / len(train_dataloader)))
    
wandb.finish()

# save model
# this function is overridden by the peft library
import time
now = time.time()
model.save_pretrained(f"custom-models/{model_name}-{ds_name}-{now}.pt")