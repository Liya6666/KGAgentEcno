import json
import evaluate
import argparse
import os
from openai import OpenAI

from IPython import embed

parser = argparse.ArgumentParser("")
parser.add_argument("--result_file", type=str, default="/Users/yehaoran/Desktop/KGAgentEcno/Graph-CoT-main/LLM/result/run_LLM_rag_results.json")
parser.add_argument("--model", type=str, default="None")
parser.add_argument("--openai_key", type=str, default="")
args = parser.parse_args()


def compute_exact_match(predictions, references):
    em_metric = evaluate.load('exact_match')
    return em_metric.compute(predictions=predictions, references=references)


def compute_bleu(predictions, references):
    bleu_metric = evaluate.load('sacrebleu')
    return bleu_metric.compute(predictions=predictions, references=references)


def compute_rouge(predictions, references):
    rouge_metric = evaluate.load('rouge')
    return rouge_metric.compute(predictions=predictions, references=references)


def GPT4score(predictions, references, questions):
    client = OpenAI(
        base_url="https://api.deepseek.com/v1",  # DeepSeek 的接口地址
        api_key="sk-dffc730848234fc3be92bf457ce88955"  # 你在 DeepSeek 平台申请的 key
    )
    # eval_prompt = "Please help me judge if the ground truth is inside the model prediction with string match.\nModel prediction: {} \nGround truth: {}. \nPlease answer Yes or No."
    eval_prompt = "Question:{} \nModel prediction: {} \nGround truth: {}. \nPlease help me judge if the model prediction is correct or not given the question and ground truth answer. Please use one word (Yes or No) to answer. Do not explain."

    res = []
    for pred, ref, question in zip(predictions, references, questions):
        x = eval_prompt.format(question, pred, ref)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a generative language model evaluator."},
                {"role": "user", "content": x},
            ],
            temperature=0.01,
            top_p=1.0,
        )
        GPT_score = response.choices[0].message.content.strip()

        # 处理API返回结果，提取Yes/No
        if 'yes' in GPT_score.lower() or 'correct' in GPT_score.lower():
            res.append(1)
        elif 'no' in GPT_score.lower() or 'incorrect' in GPT_score.lower():
            res.append(0)
        else:
            # 如果无法判断，打印返回内容并默认为0
            print(f"Unexpected response: {GPT_score}")
            res.append(0)
    return sum(res) / len(res)


def read_json(file):
    results = []
    preds = []
    gts = []
    questions = []
    with open(file) as f:
        readin = f.readlines()
        for line in readin:
            tmp = json.loads(line)
            results.append(tmp)
            preds.append(tmp['model_answer'])
            gts.append(tmp['gt_answer'])
            questions.append(tmp['question'])
    return results, preds, gts, questions


def save_results(model_name, em_score, bleu_score, rouge_score, gpt4_score):
    # 确保目录存在
    save_dir = "/Users/yehaoran/Desktop/KGAgentEcno/eval_results"
    os.makedirs(save_dir, exist_ok=True)

    # 准备结果数据
    results = {
        "model": model_name,
        "exact_match": em_score['exact_match'],
        "bleu": bleu_score['score'],
        "rouge1": rouge_score['rouge1'],
        "rouge2": rouge_score['rouge2'],
        "rougeL": rouge_score['rougeL'],
        "rougeLsum": rouge_score['rougeLsum'],
        "gpt4_score": gpt4_score
    }

    # 保存到JSON文件
    save_path = os.path.join(save_dir, "run_LLM_rag_eval.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {save_path}")
    return save_path


results, preds, gts, questions = read_json(args.result_file)
preds = [pred if pred != None else '' for pred in preds]
em_score = compute_exact_match(preds, gts)
bleu_score = compute_bleu(preds, gts)
rouge_score = compute_rouge(preds, gts)
gpt4_score = GPT4score(preds, gts, questions)

# 打印结果（保持原有输出）
print(
    f"{args.model} || EM: {em_score['exact_match']} | Bleu: {bleu_score['score']} | Rouge1: {rouge_score['rouge1']} | Rouge2: {rouge_score['rouge2']} | RougeL: {rouge_score['rougeL']} | RougeLSum: {rouge_score['rougeLsum']} | GPT4score: {gpt4_score}")

# 保存结果到文件
save_results(args.model, em_score, bleu_score, rouge_score, gpt4_score)