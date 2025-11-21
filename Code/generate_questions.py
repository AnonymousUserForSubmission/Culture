import json
from openai import OpenAI
import random
from tqdm import tqdm
import os


API_KEY = ""
URL = ""

target = "Spanish"
instructions = {
    "Factual": "Based on the context, think through all relevant cultural points step by step and generate a **factual** question. The question type can include single-choice, true/false, or fill-in-the-blank. Ensure that the question stem is clear, the options are plausible but misleading (distractors), and the answer is accurate.",
    "Conceptual": "Based on the context, think through all relevant cultural points step by step and generate a **conceptual explanation** question. The question should focus on the learner's understanding of the concepts, structures, or values behind cultural phenomena, rather than simple memorization. Suitable formats include multiple-choice or true/false questions. Ensure the question is thought-provoking and the answer is well-justified.",
    "Mislead": "Based on the context, think through all relevant cultural points step by step and generate a **misleading** question to assess whether learners can identify cultural misunderstandings, stereotypes, or biases. The question should focus on learners' critical thinking about culture—identifying which statements or behaviors reflect misunderstandings, oversimplifications, biases, or stereotypes—and guide them toward more accurate or respectful understandings. Possible formats include multiple-choice, true/false, case analysis, or short-answer questions.",
    "Multi-hop": "Based on the context, think through all relevant cultural points step by step and generate a **multi-hop reasoning** question to assess whether the learner can synthesize multiple cultural elements and understand the deeper logic or internal connections among cultural phenomena. The question should prompt learners to start from multiple information points, integrate cultural knowledge, and perform logical analysis, comparison, or generalization. Scenario-based, integrated analysis, or comparative reasoning questions are recommended."
}
# target = "中国"
# instructions = {
#     "Factual": "请根据上下文，分步骤思考所有相关文化点，生成一个【事实性】问题。题目类型可以包括单项选择题、判断题、填空题等。请确保题干清晰，"
#                "选项具有干扰性，答案准确。",
#     "Conceptual": "请根据上下文，分步骤思考所有相关文化点，生成一个【概念解释】问题。题目应聚焦于学生对文化现象背后概念、结构或价值观的理解，"
#                   "而非简单记忆。题型可以包括多项选择题、判断题等。请确保问题具有思考性，答案有明确的依据。"
#                   ,
#     "Mislead": "请基于上下文，分步骤思考所有相关文化点，生成一个【误导性】问题，"
#                "考察学生是否能辨别文化误解、刻板印象或偏见。题目应聚焦于学生对文化的批判性思考能力，能识别哪些说法或行为体现了对文化的误解、"
#                "简化、偏见或刻板印象，并引导他们提出更准确或尊重的理解。可包含选择题、判断题、案例分析题或简答题。",
#     "Multi-hop": "请根据上下文，分步骤思考所有相关文化点，生成一个【多轮推理】问题，以考察其能否综合多个文化要点，理解文化深层逻辑或文化现象"
#                  "之间的内在联系。题目应引导学生从多个信息点出发，整合文化知识，进行逻辑分析、比较或归纳。可设计情境题、综合分析题、对比推理题等。"
# }


# randomly pick 3 features for each dimension
# and generate 2 questions for each question type
# total question num: question_num * question types(4) * dims(140)
def generate_question(model_name, features_num=3, question_num=20):
    with open("knowledge-base.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    api_key = API_KEY
    url = URL
    if "gpt" in model_name:
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(
            api_key=api_key,
            base_url=url,
        )

    with open("questions.json", "r", encoding="utf-8") as f:
        generated = json.load(f)
    finished = []
    for q in generated:
        dim = q["dimension"]
        type_ = q["type_"]
        finished.append((dim, type_))
    print(f"{len(finished)} items finished.")
    output = generated
    # for file in os.listdir("questions"):
    #     finished.append(file.replace(".json", ""))
    for dim, rules_dict in tqdm(data.items(), desc=f"generating"):
        # generate 20 questions for each type
        for i in range(question_num):

            # rules = rules_dict["rules"]
            rules = [dict_["feature"] for dict_ in rules_dict]
            chosen_features = random.sample(rules, min(features_num, len(rules)))
            text = ""
            for cnt, item in enumerate(chosen_features):
                feature = item
                text += f"{cnt + 1}. {feature}\n"
            for type_, instruction in instructions.items():
                if (dim, type_) in finished:
                    finished.remove((dim, type_))
                    print(f"跳过一个{dim}, {type_}")
                    continue
                prompt = (
                    f"Task: Answer in English. {instruction}\n"
                    f"Note:\n"
                    f"1. The question should avoid explicitly mentioning cultural concepts, terminology, or characteristics, in order to effectively assess the student's understanding of cultural traits.\n"
                    f"2. A reference answer should be provided after the question.\n"
                    f"Context: '''\n{text}\n'''\n"
                    f"### Question:\n### Reference Answer:")
                # print(prompt)
                while True:
                    try:
                        completion = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {'role': 'system', 'content': f"You are an expert in {target} culture"},
                                {'role': 'user', 'content': prompt}],
                        )
                        response = completion.choices[0].message.content
                        print(f"generated a {dim}, {type_}")
                        # print(response)
                        break
                    except Exception as e:
                        continue
                epoch = {"dimension": dim, "type_": type_, "response": response, "reference": text}
                output.append(epoch)
            with open(f"questions.json", "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=4)


generate_question(model_name="qwen-max", question_num=20)
