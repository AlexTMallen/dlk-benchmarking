from argparse import ArgumentParser
from collections import namedtuple
import json
from datasets import DatasetDict, load_from_disk, Dataset
from itertools import islice
from peft import get_peft_model, LoraConfig, TaskType, PeftType
from popqa_meta_templates import templatize_ds
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
import torch
import wandb
import time

wandb.login()


parser = ArgumentParser()
parser.add_argument("--model-name", type=str, default="EleutherAI/pythia-160m")
parser.add_argument("--ds-name", type=str, default="./custom-datasets/popqa_90")
parser.add_argument("--objective", type=str, default="standard", choices=["standard", "KL+standard"])
parser.add_argument("--kl-weight", type=float, default=0.1)
parser.add_argument("--max-length", type=int, default=1024)
parser.add_argument("--pretraining-max-length", type=int, default=1024)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--n-epochs", type=int, default=2)
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
parser.add_argument("--device2", type=str, default="cuda")
parser.add_argument("--no-peft", action="store_true")
parser.add_argument("--disable-cache", action="store_true")
parser.add_argument("--target-modules", nargs="+", default=["dense_h_to_4h", "dense_4h_to_h", "query_key_value"])

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
now = time.time()
save_name = f"custom-models/{model_name}-{ds_name}-{now}.pt"

# config for wandb
cfg = vars(args)
cfg["save_name"] = save_name

### LOAD/PROCESS DATASET, AND TRAIN MODEL ###

# load dataset
ds = load_from_disk(ds_name)

ds["train"] = ds["train"].shuffle()
ds["validation"] = ds["validation"].shuffle()
ds["test"] = ds["test"].shuffle()

n_train = len(ds["train"]) if n_train == -1 else n_train
n_val = len(ds["validation"]) if n_val == -1 else n_val
n_test = len(ds["test"]) if n_test == -1 else n_test
ds = DatasetDict({
    "train": ds["train"].select(range(n_train)),
    "validation": ds["validation"].select(range(n_val)),
    "test": ds["test"].select(range(n_test))
})

# apply various templates, SOME OF WHICH FLIP THE LABEL
ds = templatize_ds(ds)

# instantiate tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def to_tensors(seq, batch_size):
    out = []
    for i in range(batch_size):
        out.append(torch.tensor(seq[i][:max_length]))
    return out


def encode_choices(examples):
    return [tokenizer.encode(cs, add_special_tokens=False, return_tensors="pt").squeeze()
             for cs in examples["choices"]]


def pad(seq, with_tok, batch_size, max_length):
    # in-place pad everything to max_length and convert to tensors
    for i in range(batch_size):
        seq[i] = [with_tok] * (max_length - len(seq[i])) + seq[i]


def tokenize_eval_examples(examples):
    # similar to tokenize_examples, but without the label
    batch_size = len(examples["text"])

    # tokenize inputs
    model_inputs = tokenizer(examples["text"])
    
    pad(model_inputs["input_ids"], tokenizer.pad_token_id, batch_size, max_length)
    pad(model_inputs["attention_mask"], 0, batch_size, max_length)

    out_dict = model_inputs
    out_dict["labels"] = torch.tensor(examples["label"])
    out_dict["true_labels"] = torch.tensor(examples["true_label"])
    out_dict["choice_ids"] = encode_choices(examples)
    out_dict["p_true"] = torch.tensor(examples["label"], dtype=torch.float32)
    return out_dict


# define templatize and tokenize functions
def tokenize_examples(examples):
    batch_size = len(examples["text"])
    print(batch_size)

    # label could be a float, representing the probability the model should assign to the statement
    targets = [choices[int(label)] for label, choices in zip(examples["label"], examples["choices"])]
    
    # tokenize inputs and targets
    inputs = tokenizer(examples["text"])
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

    pad(inputs["input_ids"], tokenizer.pad_token_id, batch_size, max_length)
    pad(inputs["attention_mask"], 0, batch_size, max_length)
    pad(labels["input_ids"], -100, batch_size, max_length)
    
    inputs["input_ids"] = to_tensors(inputs["input_ids"], batch_size)
    inputs["attention_mask"] = to_tensors(inputs["attention_mask"], batch_size)
    inputs["labels"] = to_tensors(labels["input_ids"], batch_size)
    inputs["choice_ids"] = encode_choices(examples)
    inputs["p_true"] = torch.tensor(examples["label"], dtype=torch.float32)
    print(tokenizer.decode(inputs["input_ids"][0]))
    return inputs


def get_pretraining_dataloader(num_rows=200):
    texts = []

    with open("pile/val.jsonl") as f:
        for line in islice(f, num_rows):
            texts.append(json.loads(line)["text"])

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=args.pretraining_max_length, return_tensors="pt", text_target=texts)
    encodings_ds = Dataset.from_dict(encodings)
    return DataLoader(encodings_ds, batch_size=1, shuffle=False, collate_fn=default_data_collator, pin_memory=True)


# templateize and tokenize train
train_encodings = ds["train"].map(
    tokenize_examples,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=not args.disable_cache,
    desc="Running tokenizer on dataset",
)
train_eval_encodings = ds["train"].select(range(n_val)).map(
    tokenize_eval_examples,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=not args.disable_cache,
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
pile_dataloader = get_pretraining_dataloader()

# validation and test
eval_encodings = ds["validation"].map(
    tokenize_eval_examples,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=not args.disable_cache,
    desc="Running tokenizer on dataset",
)

eval_dataloader = DataLoader(eval_encodings, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

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
if "KL" in args.objective:
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(args.device2).eval()

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

def logits_to_p_true(logits, choice_ids):
    assert choice_ids.shape[1] == 2
    assert choice_ids.shape[0] == logits.shape[0]  # batch size
    relevant_logits = torch.gather(logits[:, -1], 1, choice_ids)  # shape: (batch_size, 2)
    p_false, p_true = relevant_logits.softmax(dim=-1).unbind(dim=-1)
    return p_true


def ids_to_text(ids):
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


def eval_on_pile(n_eval=500, use_tqdm=False):
    model.eval()

    losses = []

    iterator = tqdm(pile_dataloader, total=n_eval) if use_tqdm else pile_dataloader
    for batch in islice(iterator, n_eval):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            losses.append(outputs.loss.item())
    return np.mean(losses), 2 * np.std(losses) / np.sqrt(len(losses))


def eval_model(use_tqdm=False, dataloader=eval_dataloader):
    model.eval()
    preds = []
    labels = []
    true_labels = []
    is_erroneous = []

    iterator = tqdm(dataloader) if use_tqdm else dataloader
    for batch in iterator:
        with torch.no_grad():
            choice_ids = batch.pop("choice_ids")
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits.cpu().float()

            p_true = logits_to_p_true(logits, choice_ids)

            predictions = p_true > 0.5
            labs = batch["labels"].tolist()
            true_labs = batch["true_labels"].tolist()
            is_err = [labs[i] != true_labs[i] for i in range(len(labs))]

            preds.extend(predictions)
            labels.extend(labs)
            true_labels.extend(true_labs)
            is_erroneous.extend(is_err)
    
    preds, labels, true_labels, is_erroneous = np.array(preds), np.array(labels), np.array(true_labels), np.array(is_erroneous)
    acc = accuracy_score(labels, preds)
    acc_err = accuracy_score(labels[is_erroneous], preds[is_erroneous])
    true_acc_err = accuracy_score(true_labels[is_erroneous], preds[is_erroneous])
    acc_non_err = accuracy_score(labels[~is_erroneous], preds[~is_erroneous])
            
    return namedtuple("EvalResults", ["acc", "acc_err", "true_acc_err", "acc_non_err"])(acc, acc_err, true_acc_err, acc_non_err)

eval_result = eval_model(use_tqdm=True)
print(f"Initial Acc: {eval_result.acc}, Acc on erroneous: {eval_result.acc_err}, True acc on erroneous: {eval_result.true_acc_err}, Acc on non-erroneous: {eval_result.acc_non_err}")
train_eval_result = eval_model(use_tqdm=True, dataloader=train_eval_dataloader)
print(f"Initial Train Acc: {train_eval_result.acc}, Train Acc on erroneous: {train_eval_result.acc_err}, Train Acc on non-erroneous: {train_eval_result.acc_non_err}")
pretraining_loss, pm = eval_on_pile(n_eval=len(pile_dataloader), use_tqdm=True)
print(f"Initial pretraining loss: {pretraining_loss} ± {pm}")
pretraining_loss, pm = eval_on_pile(n_eval=len(pile_dataloader), use_tqdm=True)
print(f"Initial pretraining loss (rerun): {pretraining_loss} ± {pm}")

# only the LORA parameters should be updated
learnable_parameters = [p for p in model.parameters() if p.requires_grad]
print(f"Number of learnable parameters: {len(learnable_parameters)}")
optimizer = AdamW(learnable_parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))  # adam beta2 default is 0.999

lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )


def KL(ps, base_ps):
    """Compute the KL divergence between the model logits and the base model logits
     logits: (batch_size, vocab_size) last token logits
     base_logits: (batch_size, vocab_size) last token logits
     choice_ids: (batch_size, 2) ids of the two choices
     p_true: (batch_size) probability of the true choice
    """
    base_ps = base_ps.detach().to(ps.device)

    ps = ps.clamp(1e-15, 1 - 1e-4)  # avoid numerical issues
    base_ps = base_ps.clamp(1e-15, 1 - 1e-4)
    # compute KL divergence
    kl = (ps * (ps.log() - base_ps.log())).sum(dim=-1)  # shape: (batch_size)
    return kl.mean()


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        choice_ids = batch.pop("choice_ids")
        p_true = batch.pop("p_true")
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        if "KL" in args.objective:
            device2_batch = {k: v.to(args.device2) for k, v in batch.items()}
            with torch.no_grad():
                base_outputs = base_model(**device2_batch)
            
            ps = outputs.logits[:, -1, :].type(torch.float64).softmax(dim=-1)
            base_ps = base_outputs.logits[:, -1, :].type(torch.float64).softmax(dim=-1)
            kl = None
            if args.objective == "KL+standard":
                kl = KL(ps, base_ps)
                loss = args.kl_weight * kl + outputs.loss
        elif args.objective == "standard":
            loss = outputs.loss
        else:
            raise ValueError(f"Unknown objective: {args.objective}")

        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if (step + 1) % eval_interval == 0:
            eval_result = eval_model(use_tqdm=False)
            print(f"Acc: {eval_result.acc}, Acc on erroneous: {eval_result.acc_err}, True acc on erroneous: {eval_result.true_acc_err}, Acc on non-erroneous: {eval_result.acc_non_err}")

            train_eval_result = eval_model(use_tqdm=False, dataloader=train_eval_dataloader)
            print(f"Train Acc: {train_eval_result.acc}, Train Acc on erroneous: {train_eval_result.acc_err}, Train Acc on non-erroneous: {train_eval_result.acc_non_err}")

            pretraining_loss, pm = eval_on_pile(use_tqdm=False)
            print(f"Pretraining loss: {pretraining_loss} ± {pm}")
            wandb.log({"train_acc": train_eval_result.acc, "train_acc_err": train_eval_result.acc_err, "true_train_acc_err": train_eval_result.true_acc_err, "train_acc_non_err": train_eval_result.acc_non_err,
                       "acc": eval_result.acc, "acc_err": eval_result.acc_err, "true_acc_err": eval_result.true_acc_err, "acc_non_err": eval_result.acc_non_err,
                       "train_loss": total_loss / step, "step": step, "epoch": epoch, "train_kl": kl, "pretraining_loss": pretraining_loss})

            model.train()
            
    print("Epoch {} loss: {}".format(epoch, total_loss / len(train_dataloader)))

wandb.finish()

# save model
# this function is overridden by the peft library
model.save_pretrained(save_name)