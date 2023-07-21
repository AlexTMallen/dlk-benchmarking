from peft import LoraConfig, PeftType, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "EleutherAI/pythia-6.9b"
device = "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

peft_config = LoraConfig(
    peft_type=PeftType.LORA, task_type=TaskType.CAUSAL_LM,
    inference_mode=False, target_modules=["dense_h_to_4h", "dense_4h_to_h", "query_key_value"],
    r=4, lora_alpha=32, lora_dropout=0
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model = model.to(device)

lora_model_dir = "custom-models/EleutherAI/pythia-6.9b-./custom-datasets/AkariAsai/PopQA_erroneous_multi_template_90_lying_parents-1689721964.5276177.pt"

from peft import PeftModel

lora_model = PeftModel.from_pretrained(model=model, model_id=lora_model_dir)

merged_model = lora_model.merge_and_unload()

merged_model = lora_model.merge_and_unload()
