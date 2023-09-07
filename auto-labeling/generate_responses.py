import torch
from itertools import islice
from tqdm import tqdm
import uuid
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser

VICUNA_PREFIX = "A chat between a curious human and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the human's questions."


def main(model_name, device):
    # load dataset from local jsonl
    path = "data/oasst/2023-04-12_oasst_all.messages.jsonl"
    # path = "data/alpaca_data.json"


    df = pd.read_json(path, lines=True)
    
    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": device})
    if "pythia" in model_name:
        model = model.half()
    model.eval()
    prompter_df = df[(df["role"] == "prompter") & (df["lang"] == "en")]
    prompter_df.head()
    # tokenizer.eos_token = "### End"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    print(tokenizer.eos_token_id)
    try:
        max_length = model.config.max_position_embeddings
    except AttributeError:
        try:
            max_length = tokenizer.model_max_length
        except AttributeError:
            max_length = 1024
    print("Max length:", max_length)
    results = []

    n_examples = 6000
    offset = 1000
    with torch.no_grad():
        shuffled_df = prompter_df.sample(frac=1, random_state=633)
        for i, row in tqdm(enumerate(shuffled_df.iloc[offset:offset + n_examples].iloc), total=n_examples):
            print(f"\n\n\n####################################  {i}  ##############################################\n\n")
            pid = row["parent_id"]
    
            parent_text = row["text"]
            prefix = {"keyfan/bloomz-rlhf": VICUNA_PREFIX + " ",
                    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": "",
                    "lmsys/vicuna-7b-v1.5": VICUNA_PREFIX + "\n\n"}[model_name]
            
            prompter_template = {"keyfan/bloomz-rlhf": "USER: {}\n", "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": "<|prompter|>{}<|endoftext|>",
                            "lmsys/vicuna-7b-v1.5": "USER: {}</s>\n"}[model_name]
            assistant_prefix = {"keyfan/bloomz-rlhf": "ASSISTANT:", "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5": "<|assistant|>",
                                "lmsys/vicuna-7b-v1.5": "ASSISTANT:"}[model_name]

            role_text = prompter_template if row["role"] == "prompter" else assistant_prefix
            texts = [role_text + row["text"]]
            prev_messages = [row["text"]]
            while not pd.isnull(pid):
                parent = df[df["message_id"] == pid].iloc[0]
                role_text = prompter_template.format(parent["text"]) if parent["role"] == "prompter" \
                    else assistant_prefix + " " + parent["text"]  # TODO: this adds an extra space for OASST
                texts.append(role_text)
                pid = parent["parent_id"]
                prev_messages.append(parent["text"])
            

            texts.reverse()
            prev_messages.reverse()
            prompt = prefix + "".join(texts) + assistant_prefix

            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            if len(inputs["input_ids"][0]) > max_length - 256:
                print("Skipping because too long")
                continue
            
            generate_kwargs = {"max_length": max_length, "temperature": 1.0, "repetition_penalty": 1.2, "do_sample": True, "top_p": 0.95}
            gen_kwargs_str = ";".join([f"{k}={v}" for k, v in generate_kwargs.items()])

            model_output = model.generate(
                inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                eos_token_id=tokenizer.eos_token_id,
                **generate_kwargs
            )
            transcript = tokenizer.decode(model_output[0])
            response = transcript.split(assistant_prefix)[-1].lstrip().removesuffix(tokenizer.eos_token)
            print()
            print(prev_messages)
            print(response)
            results.append({
                "message_id": str(uuid.uuid4()),
                "prompt": prompt,
                "assistant_text": response,
                "parent_id": row["message_id"],
                "parent_text": parent_text,
                "prev_messages": prev_messages,
                "role": "assistant",
                "synthetic": True,
                "model": f"{model_name}({gen_kwargs_str})"
            })

            step = 50
            if len(results) % step == 0 or i == n_examples - 1:
                results_df = pd.DataFrame.from_dict(results)
                model_str = model_name.replace("/", "_")
                results_df.to_json(f"data/oasst/generated_responses/{len(results)}_transcripts_{model_str}({gen_kwargs_str}).json", orient="records")
                prev_file = f"data/oasst/generated_responses/{len(results) - step}_transcripts_{model_str}({gen_kwargs_str}).json"
                if os.path.exists(prev_file):
                    os.remove(prev_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args.model_name, args.device)
