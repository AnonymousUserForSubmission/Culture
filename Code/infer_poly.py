import os
import json
import torch
from tqdm import tqdm
import argparse

# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

# def load_model(base_model_path):
#     model = AutoModelForCausalLM.from_pretrained(
#         base_model_path,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         trust_remote_code=True
#     )
#     model.eval()
#     return model

def run(model, tokenizer, dataset, output_dir, max_new_tokens, system_prompt):
    os.makedirs(output_dir, exist_ok=True)

    # kwargs = {"do_sample": False, "num_beams": 4, "max_new_tokens": max_new_tokens, "early_stopping": True, "eos_token_id": 2}
    # pipeline_ins = pipeline(Tasks.text_generation, model=model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    for idx, entry in enumerate(tqdm(dataset, desc="Running Baseline Inference")):
        out_path = os.path.join(output_dir, f"{idx:05d}.json")
        if os.path.exists(out_path):
            continue

        question = entry.get("question", "")
        full_prompt = "<|system|>\n" + f"{system_prompt}\n" + "<|user|>\n" + f"{question}\n" + "<|assistant|>\n"

        # response = pipeline_ins(full_prompt, **kwargs)
        inputs = tokenizer(full_prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = {k: v for k, v in entry.items() if k != "question"}
        result["question"] = question
        result["response"] = response
        result["system_prompt"] = system_prompt

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default="", help="Optional system prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()


    with open(args.input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # polylm_model_id = args.base_model
    model_name = args.base_model
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                 trust_remote_code=True, bf16=True).eval()

    model.generation_config = GenerationConfig.from_pretrained(model_name,
                                                               trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    run(model, tokenizer, dataset, args.output_dir, args.max_new_tokens, args.system_prompt)
