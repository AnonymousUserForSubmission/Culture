import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse


def load_model(base_model_path):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return model

def run(model, tokenizer, dataset, output_dir, max_new_tokens, system_prompt):
    os.makedirs(output_dir, exist_ok=True)
    for idx, entry in enumerate(tqdm(dataset, desc="Running Baseline Inference")):
        out_path = os.path.join(output_dir, f"{idx:05d}.json")
        if os.path.exists(out_path):
            continue

        question = entry.get("question", "")
        # full_prompt = system_prompt + question

        full_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # response = output_text[len(full_prompt):].strip()
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

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    model = load_model(args.base_model)
    run(model, tokenizer, dataset, args.output_dir, args.max_new_tokens, args.system_prompt)
