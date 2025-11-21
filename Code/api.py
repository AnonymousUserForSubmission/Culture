import os
from openai import OpenAI
import argparse
import json
from tqdm import tqdm
import time

def inference(system, question, enable_thinking):
    client = OpenAI(
        api_key="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen3-8b",
        messages=[
            # {"role": "system", "content": system + "Answer directly. No <think> tags. No reasoning."},
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        stream=True,
        # if enable_thinking:
        extra_body={
            "enable_thinking": enable_thinking
        }
    )

    full_content = ""
    full_reasoning = ""
    for chunk in completion:
        print(chunk.choices[0].delta)
        if chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content

    # return full_content

            # print(chunk.choices[0].delta.content)
        if enable_thinking:
            if chunk.choices[0].delta.reasoning_content:
                full_reasoning += chunk.choices[0].delta.reasoning_content
        else:
            full_reasoning = "NO_THINKING"
            # print(chunk.choices[0].delta.content)
    # print(f"完整内容为：{full_content}")

    # print(completion.choices[0].message.content)
    # return completion.choices[0].message.content
    return full_content, full_reasoning

def run(dataset, output_dir, max_new_tokens, system_prompt, enable_thinking):
    os.makedirs(output_dir, exist_ok=True)
    for idx, entry in enumerate(tqdm(dataset, desc="Running Baseline Inference")):
        out_path = os.path.join(output_dir, f"{idx:05d}.json")
        if os.path.exists(out_path):
            continue

        question = entry.get("question", "")
        while(True):
            try:
                content, reasoning = inference(system_prompt, question, enable_thinking)
                break
            except Exception as e:
                print(e)
                time.sleep(1)

        # content = inference(system_prompt, question, enable_thinking)
        # except Exception:
        #     content = "Generation Failed"

        result = {k: v for k, v in entry.items() if k != "question"}
        result["question"] = question
        # result["response"] = response
        result["response"] = f"ANSWER:::{content}, reason: {reasoning}"
        # result["response"] = f"ANSWER:::{content}"
        result["system_prompt"] = system_prompt

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default="", help="Optional system prompt")
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--enable_thinking", type=bool, default=False)
    args = parser.parse_args()

    # tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)

    with open(args.input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # model = load_model(args.base_model)
    print("THINK: ", args.enable_thinking)
    print(type(args.enable_thinking))
    run(dataset, args.output_dir, args.max_new_tokens, args.system_prompt, args.enable_thinking)


