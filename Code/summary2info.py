import json
import os
import re

LLM = "qwen-plus"


def strip_prefix_if_present(s: str) -> str:
    """
    If s starts with either
      1) one or more digits + '.'  (e.g. "12.")
      2) '特点' + Chinese numeral 一–九 + ':' (e.g. "特点五:")
    (plus any whitespace),
    strip that off and return the rest. Otherwise return s unchanged.
    """
    PREFIX_RE = re.compile(r'^(?:\d+\. |特点[一二三四五六七八九]:|特点[一二三四五六七八九]：)\s*')
    m = PREFIX_RE.match(s)
    if m:
        # m.end() is the index right after the matched prefix
        return s[m.end():]
    return s


# parse summary and split into cultural info
# input: candidates{url, summary} and source webpage
# return [{characteristic, original information}]
def extract_from_candidates(candidates, webpage):
    output = []
    for candidate in candidates:
        candidate = candidate.replace("*", "").strip()
        if "信息来源" in candidate:
            try:
                # title = candidate.split("特点描述")[0]
                # info = candidate.split("特点描述")[1]
                # desc = info.split("信息来源")[0]
                # source = info.split("信息来源")[1].replace("：", "") + f"===from {webpage}"
                feature = candidate.split("信息来源")[0].replace("#", "").split("特点描述")[1]
                source = candidate.split("信息来源")[1].replace("：", "").replace(":", "").replace("#", "") + f"===from {webpage}"

            except Exception:
                continue
        # elif "features description" in candidate.lower():
        #     try:
        #         title = candidate.lower().split("features description")[0]
        #         info = candidate.lower().split("features description")[1]
        #         desc = info.split("information source")[0]
        #         source = info.split("information source")[1].replace("：", "") + f"===from {webpage}"
        #     except Exception:
        #         continue
        # elif "characteristics description" in candidate.lower():
        #     try:
        #         title = candidate.lower().split("characteristics description")[0]
        #         info = candidate.lower().split("characteristics description")[1]
        #         desc = info.split("information source")[0]
        #         source = info.split("information source")[1].replace("：", "") + f"===from {webpage}"
        #     except Exception:
        #         continue
        else:
            continue

        # feature = title + desc
        feature = strip_prefix_if_present(feature.replace("：", ":").replace("- ", "").replace("\n", "").replace("  ", "")).replace("#", "")
        if len(feature) < 20:
            continue
        epoch = {"feature": feature, "original": source}
        output.append(epoch)

    return output


# return: culture_info and source
def summary_to_info(dict_):
    try:
        text = dict_["response"]
    except Exception:
        text = dict_["test_response"]
    url = dict_["url"]

    if "access denied" in text or "一般知识" in text or "二进制" in text:
        # llm might provide some info
        source = LLM
    elif "一般" in text and "文档" in text or "一般" in text and "文章" in text:
        source = LLM
    else:
        source = url

    candidates = text.split("###")
    if len(candidates) > 1:
        if "*" not in candidates[0]:
            candidates = candidates[1:]
        # try:
        infos = extract_from_candidates(candidates, source)
        # if len(infos) == 0:
        #     print(f"nothing to extract from {text}")
        # except Exception:
        #     print(text)
        return infos
    else:
        return None


# build cultural-base containing:
def build_info_base(parent_dir="summary"):
    output = {}
    dims = []
    for file in os.listdir("dimensions"):
        dim_1 = file.split(".")[0]
        # if dim_1 in ["Personal_Choices_and_Habits", "Social_Relationship_and_Structures", "Values_and_Beliefs"]:
        #     continue
        with open(os.path.join("dimensions", file), "r") as f:
            data = json.load(f)
        for dim_2, dim_3_list in data.items():
            for dim_3 in dim_3_list:
                if (dim_1, dim_2, dim_3) not in dims:
                    dims.append((dim_1, dim_2, dim_3))
    print(f"{len(dims)} dims in total")
    for dim_1, dim_2, dim_3 in dims:
        if os.path.isfile(f"{parent_dir}/{dim_1}-{dim_2}-{dim_3}.json"):
            with open(f"{parent_dir}/{dim_1}-{dim_2}-{dim_3}.json", "r") as f:
                data = json.load(f)
        else:
            try:
                with open(f"{parent_dir}/{dim_1}_{dim_2}_{dim_3}.json", "r") as f:
                    data = json.load(f)
            except Exception:
                print(f"lack of {dim_1}-{dim_2}-{dim_3}")
                continue

        current_dim = dim_1 + "-" + dim_2 + "-" + dim_3
        for item in data:
            infos = summary_to_info(item)

            if infos is None:
                # print(f"{dim_1}-{dim_2}-{dim_3} has no info")
                continue
            # if len(infos) == 0:
                # print(f"{dim_1}-{dim_2}-{dim_3} was not parsed correctly")
            else:
                if current_dim not in output:
                    output[current_dim] = infos
                else:
                    output[current_dim].extend(infos)
    for dim, infos in output.items():
        if len(infos) < 1:
            print(f"no info for {dim}")
    print(f"{sum(len(values) for key, values in output.items())} rules in total")
    with open("knowledge-base.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


build_info_base()