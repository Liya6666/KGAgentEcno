import os
import re
import tqdm
import time
import logging
import jsonlines
import argparse
import requests  # 使用 requests 库替换 OpenAI 的客户端

from typing import List, Dict, Any
import asyncio

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def clean_str(string):
    pattern = re.compile(r'^\d+\. ', flags=re.MULTILINE)
    string = pattern.sub('', string)
    return string.strip()


def dispatch_deepseek_requests(
        args,
        messages_list: List[List[Dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
) -> List[str]:
    """Dispatches requests to DeepSeek-chat API.

    Args:
        messages_list: List of messages to be sent to DeepSeek-chat API.
        model: DeepSeek-chat model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from DeepSeek-chat API.
    """
    headers = {
        "Authorization": f"Bearer {args.deepseek_key}",  # 使用 DeepSeek 的 API 密钥
        "Content-Type": "application/json"
    }

    # 正确的 DeepSeek Chat Completions 接口（不要带 /v1）
    url = "https://api.deepseek.com/chat/completions"

    async_responses = []
    for x in messages_list:
        payload = {
            "model": model,             # 例如 "deepseek-chat"
            "messages": x,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        # 按你原有的“收集再统一 .json()”的同步风格，不改逻辑
        async_responses.append(
            requests.post(url, json=payload, headers=headers)
        )

    # 保持你原有的返回风格（不加额外容错，尽量不改逻辑）
    return [response.json() for response in async_responses]


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--dataset", type=str, default="amazon")
    # 默认改成 deepseek-chat；并放宽可选模型范围以支持 deepseek
    parser.add_argument("--gpt_version", type=str, default="deepseek-chat")
    parser.add_argument("--data_file", type=str, default="/Users/yehaoran/Desktop/KGAgentEcno/Graph-CoT-main/data/processed_data/amazon/new_data.json")
    parser.add_argument("--save_file", type=str, default="/Users/yehaoran/Desktop/KGAgentEcno/Graph-CoT-main/GPT/results/run_GPT_results.json")
    parser.add_argument("--deepseek_key", type=str, default="sk-dffc730848234fc3be92bf457ce88955")  # 保留参数位置，不回显你的真实密钥
    args = parser.parse_args()

    # 放宽 assert，允许 deepseek 的模型名
    assert args.gpt_version in ['deepseek-chat', 'deepseek-reasoner', 'gpt-3.5-turbo', 'gpt-4']

    file_path = args.data_file
    with open(file_path, 'r') as f:
        contents = []
        for item in jsonlines.Reader(f):
            contents.append(item)

    system_message = "You are an AI assistant to answer questions. Please use your own knowledge to answer the questions. If you do not know the answer, please guess a most probable answer. Only include the answer in your response. Do not explain."
    query_messages = []
    for item in contents:
        message = item["question"]
        query_messages.append([
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ])
    generated_text = []

    for i in tqdm.trange(0, len(query_messages), 20):
        try:
            response = dispatch_deepseek_requests(
                args,
                messages_list=query_messages[i:i + 20],
                model=args.gpt_version,
                temperature=0.01,
                max_tokens=2048,
                top_p=1.0,
            )
            time.sleep(5)  # 请求间隔 5 秒（你要求）
        except:
            print("rate limit exceeded, sleep for 5 seconds")
            time.sleep(5)  # Rate Limit 也改成 5 秒（你要求）
            response = dispatch_deepseek_requests(
                args,
                messages_list=query_messages[i:i + 20],
                model=args.gpt_version,
                temperature=0.01,
                max_tokens=2048,
                top_p=1.0,
            )
        for j in range(len(response)):
            generated_text.append({
                "qid": contents[i + j]["qid"],  # 添加问题ID
                "question": contents[i + j]["question"],
                "model_answer": response[j]["choices"][0]["message"]["content"],
                "gt_answer": contents[i + j]["answer"]
            })
    # 打印首条与数量；如果为空不会崩
    if generated_text:
        print(generated_text[0], len(generated_text))
    else:
        print("No generated text.")

    output_file_path = args.save_file

    parent_folder = os.path.dirname(output_file_path)
    parent_parent_folder = os.path.dirname(parent_folder)
    if not os.path.exists(parent_parent_folder):
        os.mkdir(parent_parent_folder)
    if not os.path.exists(parent_folder):
        os.mkdir(parent_folder)

    with jsonlines.open(output_file_path, 'w') as writer:
        for row in generated_text:
            writer.write(row)


if __name__ == '__main__':
    main()
