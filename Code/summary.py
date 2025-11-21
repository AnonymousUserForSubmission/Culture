import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
import time
# import gc
# from torch.amp import autocast
from openai import OpenAI

'''
    Summary the crawled texts to get cultural knowledge instances
'''
qwen_api_key = ""
qwen_url = ""

# with open("finished.json", "r", encoding="utf-8") as f:
#     finished = json.load(f)
#
# with open("failed.json", "r", encoding="utf-8") as f:
#     failed = json.load(f)
finished = []
failed = []

api_key = qwen_api_key
url = qwen_url
client = OpenAI(
    api_key=api_key,
    base_url=url,
)

# def model_inference(model_name, model, tokenizer, prompt, content):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # messages = [
    #     {"role": "system", "content": content},
    #     {"role": "user", "content": prompt}
    # ]
    #
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    #
    # generated_ids = model.generate(
    #     **model_inputs,
    #     max_new_tokens= 512
    # )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]
    # with torch.no_grad():
    #     # with autocast(device_type=device):
    #     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    #
    # print(response)


# Max size in bytes
max_words = 120000

def model_inference(model_name, prompt, content):
    # truncated_prompt = truncate_input_by_size(prompt, max_size)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': content},
            {'role': 'user', 'content': prompt}],
    )
    response = completion.choices[0].message.content
    print(response)

    return response


def run_test(model_name, target="Chinese"):
    with open(f"finished_attempt2.json", "r", encoding="utf-8") as f:
        finished = json.load(f)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype="auto",
    #     device_map="auto"
    #
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    identity = "你是一个善于信息提取与总结的语言模型。"
    for level_1 in os.listdir("text"):
        # if level_1 in ["Personal_Choices_and_Habits", "Social_Relationship_and_Structures", "Values_and_Beliefs"]:
        #     continue
        for level_2 in os.listdir(os.path.join("text", level_1)):
            for level_3 in tqdm.tqdm(os.listdir(os.path.join("text", level_1, level_2)), desc=f"processing {level_1}_{level_2}"):
                if f"{level_1}_{level_2}_{level_3}" in finished:
                    print(f"skipping {level_1}_{level_2}_{level_3}")
                    continue
                data_path = os.path.join(f"text", level_1, level_2, level_3)
                with open(data_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                # if len(content) < 3:
                #     failed.append(f"{level_1}_{level_2}_{level_3}")
                #     print(f"lack of {level_1}_{level_2}_{level_3}")
                #     continue
                output = []
                dim = level_3.replace(".json", "")
                for item in tqdm.tqdm(content, desc=f"{level_1}_{level_2}_{level_3}"):
                    url = item["url"]
                    text = item["text"]

                    # len_restart = False
                    restart_length = max_words
                    while True:  # 内层控制重试
                        cut_off_length = restart_length
                        truncated_text = text[:cut_off_length]
                        try:
                            # prompt = (f"我将给出一个网页文本，请你从中提取出关于中国文化中"
                            #           f"“{dim}” 的主要特点与内容，并用清晰的标题列出这些特点。每一条特点都用“### 标题”开始，下面分点写出："
                            #           "- 特点描述：\n- 信息来源：（引用英文原文，并尽量指出出处段落）内容条理清晰、逻辑紧凑。"
                            #           f"如果信息不足以支持某个特点，请不要编造内容。文章如下：===\n{truncated_text}\n===")
                            prompt = (
                                f"I will provide a webpage text. Please extract the main features and content related to "
                                f"'{dim}' in Chinese culture from it, and list these features with clear headings. "
                                f"Start each feature with '### Title', and then explain with the following points:\n"
                                f"- Feature description:\n- Source of information: (quote the original English text and, "
                                f"if possible, indicate the paragraph or section)\n"
                                f"The content should be well-organized and logically coherent. "
                                f"If the information is insufficient to support a certain feature, please do not fabricate content.\n"
                                f"The article is as follows: ===\n{truncated_text}\n===")

                            response = model_inference(model_name, prompt, identity)
                            break
                        except Exception as e:
                            print(e)
                            if e.code == "invalid_parameter_error":
                                time.sleep(2)
                                restart_length -= 10000
                                print(f"Reset cutoff length to {restart_length}")
                            elif e.code != "data_inspection_failed":
                                time.sleep(2)
                            else:
                                response = "Data inspection failed."
                                break
                    epoch = {"url": url, "response": response}
                    output.append(epoch)

                with open(f"summary/{level_1}_{level_2}_{dim}.json", "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=4)

                finished.append(f"{level_1}_{level_2}_{level_3}")
                with open(f"finished_attempt2.json", "w", encoding="utf-8") as f:
                    json.dump(finished, f, ensure_ascii=False, indent=4)
                with open(f"failed_attempt2.json", "w", encoding="utf-8") as f:
                    json.dump(failed, f, ensure_ascii=False, indent=4)


# model_path = "../Qwen/Qwen2___5-14B-Instruct"
model_path = "qwen-plus"
model_name = model_path.split("/")[-1].replace("___", "-")

run_test(model_path)

# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(output, f, ensure_ascii=False, indent=4)
