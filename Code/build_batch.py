import json
import os

def build_test_batch():
    output_path = "test_v3_english_batch.jsonl"
    output = []
    custom_id = 1
    id2answer = {}
    id2file = {}
    system = "You are a cultural assistant. Please answer the question after ::: on a new line. Do not include anything after ::: except the final answer."
    # system = "你是一名文化助理。请在 ::: 后换行回答问题。除了最终答案，不要包含 ::: 后的任何其他内容。"


    with open(os.path.join("test_data", "English.json"), "r", encoding="utf-8") as f:
        questions = json.load(f)

    for input_data in questions:
        question = input_data["question"]
        answer = input_data["answer"]

        epoch = {"custom_id": str(custom_id), "method": "POST", "url": "/v1/chat/completions",
                 "body": {"model": "deepseek-v3", "messages": [{"role": "system", "content": system},
                                                               {"role": "user", "content": question}]}}
        output.append(epoch)
        custom_id += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for epoch in output:
            f.write(json.dumps(epoch, ensure_ascii=False) + "\n")


build_test_batch()