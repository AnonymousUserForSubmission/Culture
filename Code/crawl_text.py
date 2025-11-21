from web_crawler import *
import os
import json
import string
import re
from tqdm import tqdm

'''
    Crawl cultural-specific texts from Internet
'''
# if a proxy is needed
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# finished dimensions and failed dimensions
failed = []
finished = []
if len(finished) == 0:
    with open("crawl_finished.json", "r", encoding="utf-8") as f:
        finished = json.load(f)
if len(failed) == 0:
    with open("crawl_failed.json", "r", encoding="utf-8") as f:
        failed = json.load(f)

# target culture name
# target = "Spanish"
# target = "Chinese"

#
print(f"{finished} is finished")
def do_crawl():
    for level_1 in os.listdir("dimensions"):
        level_1 = level_1.replace(".json", "")
        # found = False
        # for labels in finished:
        #     if level_1 in labels:
        #         found = True
        # if found:
        #     continue
        if not os.path.exists(os.path.join("text", level_1)):
            os.makedirs(os.path.join("text", level_1))
        if not os.path.exists(os.path.join("text", level_1)):
            os.makedirs(os.path.join("text", level_1))

        with open(os.path.join("dimensions", f"{level_1}.json"), "r", encoding="utf-8") as f:
            l1_data = json.load(f)
        print(f"{sum(len(v) for v in l1_data.values())} dims in {level_1}")

        for level_2, level_3_list in l1_data.items():
            if not os.path.exists(os.path.join("text", level_1, level_2)):
                os.makedirs(os.path.join("text", level_1, level_2))
            if not os.path.exists(os.path.join("text", level_1, level_2)):
                os.makedirs(os.path.join("text", level_1, level_2))

            for level_3 in tqdm(level_3_list, desc=f"processing subdims of {level_2}"):
                if f"{level_1}-{level_2}-{level_3}" in finished:
                    continue
                query = f"中国文化中的{level_3}"
                crawled = search_and_crawl(query, num_results=5)
                if len(crawled) == 0:
                    failed.append(f"{level_1}-{level_2}-{level_3}")
                result = []
                for url, html in crawled.items():
                    try:
                        crawled_text = extract_text(html)
                    except Exception as e:
                        continue
                    result.append({"url": url, "text": crawled_text})
                with open(os.path.join("text", level_1, level_2, f"{level_3}.json"), "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)

                finished.append(f"{level_1}-{level_2}-{level_3}")
            with open("crawl_finished.json", "w", encoding="utf-8") as f:
                json.dump(finished, f, ensure_ascii=False, indent=4)

            with open("crawl_failed.json", "w", encoding="utf-8") as f:
                json.dump(failed, f, ensure_ascii=False, indent=4)


def do_failed_crawl():
    with open("crawl_failed.json", "r", encoding="utf-8") as f:
        failed = json.load(f)
    still_failed = []
    for item in failed:
        level_1, level_2, level_3 = item.split("-")
        # query = f"{level_3} in {target} culture"
        query = f"{level_3}"
        crawled = search_and_crawl(query, num_results=5)
        if len(crawled) == 0:
            print(f"still failed: {level_1}_{level_2}_{level_3}")
            still_failed.append(f"{level_1}-{level_2}-{level_3}")
        result = []
        for url, html in crawled.items():
            result.append({"url": url, "text": extract_text(html)})
        with open(os.path.join("text", level_1, level_2, f"{level_3}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    with open("crawl_failed.json", "w", encoding="utf-8") as f:
        json.dump(still_failed, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    # do_crawl()
    do_failed_crawl()