import os
import json
import torch
from tqdm import tqdm
import argparse

# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# from modelscope import AutoModelForCausalLM, AutoTokenizer
# from transformers  import pipeline
# from modelscope.utils.constant import Tasks
from transformers import AutoModelForCausalLM, AutoTokenizer

# def load_model(base_model_path):
#     model = AutoModelForCausalLM.from_pretrained(
#         base_model_path,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         trust_remote_code=True
#     )
#     model.eval()
#     return model

def run(model_id, dataset, output_dir, max_new_tokens, system_prompt):
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        # revision=revision,
        device_map="auto",
        torch_dtype=torch.float16,
        # max_new_tokens=max_new_tokens,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        # revision=revision
    )

    # pipeline_ins = pipeline(
    #     task=Tasks.text_generation,
    #     model=model,
    #     tokenizer=tokenizer
    # )
    kwargs = {"torch_dtype":torch.float16, "do_sample": False, "num_beams": 4, "max_new_tokens": max_new_tokens, "early_stopping": True, "eos_token_id": 2}
    # pipeline_ins = pipeline(Tasks.text_generation, model=model_id)


    for idx, entry in enumerate(tqdm(dataset, desc="Running Baseline Inference")):
        out_path = os.path.join(output_dir, f"{idx:05d}.json")
        if os.path.exists(out_path):
            continue

        question = entry.get("question", "")
        full_prompt = "<|system|>\n" + f"{system_prompt}\n" + "<|user|>\n" + f"{question}\n" + "<|assistant|>\n"

        # response = pipeline_ins(full_prompt, **kwargs)['text']
        # response = pipeline_ins({"text": full_prompt}, **kwargs)['text']

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        generate_kwargs = {
            "do_sample": False,
            "num_beams": 4,
            "max_new_tokens": max_new_tokens,
            "early_stopping": True,
            "eos_token_id": tokenizer.eos_token_id or 2,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)

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

    # git clone https://github.com/modelscope/modelscope
    # cd modelscope
    # pip install .


    # polylm_13b_model_id = 'iic/nlp_polylm_13b_text_generation'

    # polylm_model_id = args.base_model
    # model_name = "/".join(args.base_model.split("/")[-2])
    # model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto",
    #                                              revision='master', trust_remote_code=True, bf16=True).eval()
    #
    # model.generation_config = GenerationConfig.from_pretrained(model_name,
    #                                                            revision='master', trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=v,
    #                                           trust_remote_code=True)
    run(args.base_model, dataset, args.output_dir, args.max_new_tokens, args.system_prompt)
