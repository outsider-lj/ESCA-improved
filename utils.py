import pickle
import numpy as np
import random
import torch
import os
import sys
import time
import openai
import sqlite3
import faiss
TMP_DIR = {
    'esc': './tmp/esc',
    'cima': './tmp/cima',
    'cb': './tmp/cb',
}

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
# 策略分类映射
# Knowledge_Mapping = {
#     "information-related": ["self-disclosure", "information", "providing suggestions"],
#     "emotion-related": ["affirmation and reassurance", "reflection of feelings"],
#     "context-related": ["restatement or paraphrasing", "question", "others"]
# }
Knowledge_Mapping = {
    "information-related": ["self-disclosure", "information", "providing suggestions","affirmation and reassurance"],
    "context-related": ["restatement or paraphrasing","reflection of feelings", "question", "others"]
}
# 根据策略确定类别
def get_strategy_category(strategy):
    for cat, strategies in Knowledge_Mapping.items():
        if strategy.lower() in strategies:
            return cat
    return None

# 从 SQLite + FAISS 检索 knowledge
def search_faiss_and_fetch_text(encoder, dial_history, index_path, db_path, top_k=2):
    # 加载索引
    index = faiss.read_index(index_path)

    # 编码查询
    if isinstance(dial_history,list):
        # user_query = " ".join([f"{t['speaker']}: {t['text']}" for t in dial_history])
        user_query = " ".join([f"{t['role']}: {t['content']}" for t in dial_history])
    else:
        user_query=dial_history
    query_vector = encoder.encode([user_query])
    query_vector = np.array(query_vector).astype("float32")

    # 检索 top_k
    distances, indices = index.search(query_vector, top_k)
    # 查询 SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    results = []
    for idx in indices[0]:
        cursor.execute("SELECT context,knowledge FROM knowledge WHERE faiss_index=?", (int(idx),))
        row = cursor.fetchone()
        if row:
            context=row[0]
            knowledge=row[1]
            results.append([context,knowledge])
    conn.close()

    return results

def search_faiss_and_fetch_text_for_state(encoder, dial_history, index_path, db_path, top_k):
    print("search state example",flush=True)
    # 加载索引
    index = faiss.read_index(index_path)

    # 编码查询
    if isinstance(dial_history,list):
        # user_query = " ".join([f"{t['speaker']}: {t['text']}" for t in dial_history])
        user_query = " ".join([f"{t['role']}: {t['content']}" for t in dial_history])
    else:
        user_query=dial_history
    query_vector = encoder.encode([user_query])
    query_vector = np.array(query_vector).astype("float32")

    # 检索 top_k
    distances, indices = index.search(query_vector, top_k)

    # 查询 SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    results_ebs = []
    results_trust=[]
    for idx in indices[0]:
        cursor.execute("SELECT context,emotion,trust,behavior,stage,reliability,competence,understandability FROM knowledge WHERE faiss_index=?", (int(idx),))
        row = cursor.fetchone()
        if row:
            context=row[0]
            emotion=row[1]
            trust=row[2]
            behavior=row[3]
            stage=row[4]
            reliability=row[5]
            competence=row[6]
            understandability=row[7]
            result_ebs=str("Input: "+context+" Output: negative emotional intensity: "+emotion+" behavior: "+behavior+"stage of change: "+stage)
            result_trust=str("Input: "+context+" Output: reliablity: "+reliability+" response competence: "+competence+ " percieved understandability: "+understandability)
            results_ebs.append(result_ebs)
            results_trust.append(result_trust)
    conn.close()
    return results_ebs,results_trust

def compute_bleu4(preds, refs):
    """
    计算 BLEU-4 分数（每对句子分别计算后取平均）
    preds: list of str（预测句子）
    refs: list of str（参考句子）
    """
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    for pred, ref in zip(preds, refs):
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        bleu_scores.append(bleu)
    return sum(bleu_scores) / len(bleu_scores)


def compute_distinct2(sentences):
    """
    计算 Distinct-2 分数（所有句子的总和）
    sentences: list of str
    """
    bigram_counter = Counter()
    total_bigrams = 0

    for sentence in sentences:
        tokens = sentence.strip().split()
        bigrams = list(zip(tokens, tokens[1:]))
        bigram_counter.update(bigrams)
        total_bigrams += len(bigrams)

    if total_bigrams == 0:
        return 0.0
    return len(bigram_counter) / total_bigrams

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def load_dataset(data_name):
    dataset = {'train':[], 'test':[], 'valid':[]}
    for key in dataset:
        with open("./data/train_data/%s-%s.txt"%(data_name, key),'r') as infile:
            for line in infile:
                dataset[key].append(eval(line.strip('\n')))
    return dataset


def set_cuda(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    devices_id = [int(device_id) for device_id in args.gpu.split()]
    device = (
        torch.device("cuda:{}".format(str(devices_id[0])))
        if use_cuda
        else torch.device("cpu")
    )
    return device, devices_id
def query_chat_ai_model( messages: str, model: str , max_tokens: int = 128, temperature: float = 0, n: int = 1):
    client = openai.OpenAI(
        api_key="",
        base_url="https://api.deepseek.com"#"https://chat-ai.academiccloud.de/v1",
    )

    def _single_request(max_retries=3):
        """发送一次 API 请求 (n=1)，返回单个响应文本"""
        extra_kwargs = {}
        if "deepseek" in model.lower():
            extra_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        for attempt in range(max_retries):
            try:
                completions = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                    temperature=temperature,
                    **extra_kwargs,
                )
                return completions.choices[0].message.content.strip()
            except Exception as e:
                print(f"query_chat_ai_model 错误 (尝试 {attempt+1}/{max_retries}): {type(e).__name__}: {e}", flush=True)
                if isinstance(e, (openai.AuthenticationError, openai.NotFoundError, openai.BadRequestError)):
                    print(f"不可恢复错误，放弃重试", flush=True)
                    return ""
                elif isinstance(e, openai.RateLimitError):
                    print("请求频率超限，等待后重试...", flush=True)
                    time.sleep(10)
                else:
                    print("网络或其他错误，等待后重试...", flush=True)
                    time.sleep(5)
        print(f"请求失败超过 {max_retries} 次，返回空", flush=True)
        return ""


    if "deepseek" in model.lower() and n > 1:
        print(f"[query_chat_ai] DeepSeek 不支持 n>1，循环 {n} 次调用", flush=True)
        return [_single_request() for _ in range(n)]

    if n == 1:
        return _single_request()

    for attempt in range(3):
        try:
            print(model, flush=True)
            completions = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                n=n,
                stop=None,
                temperature=temperature,
            )
            return [choice.message.content.strip() for choice in completions.choices]
        except Exception as e:
            print(f"query_chat_ai_model 错误 (尝试 {attempt+1}/3): {type(e).__name__}: {e}", flush=True)
            if isinstance(e, (openai.AuthenticationError, openai.NotFoundError, openai.BadRequestError)):
                print(f"不可恢复错误，放弃重试", flush=True)
                return ["" for _ in range(n)]
            elif isinstance(e, openai.RateLimitError):
                print("请求频率超限，等待后重试...", flush=True)
                time.sleep(10)
            else:
                print("网络或其他错误，等待后重试...", flush=True)
                time.sleep(5)
    print(f"请求失败超过 3 次，返回空列表", flush=True)
    return ["" for _ in range(n)]

import re

def extract_summary(text):
    if text is None:
        return ""

        # 去掉首尾空白
    s = text.strip()

    # 去除外围引号
    s = re.sub(r'^[\'"“”‘’]+', '', s)
    s = re.sub(r'[\'"“”‘’]+$', '', s)

    # 保留 summary 后的内容
    m = re.search(r'(?i)summary\s*[:：]', s)
    if m:
        s = s[m.end():].lstrip()

    # 删除 Note: / NOTE: / note:
    s = re.sub(r'(?i)note\s*[:：].*$', '', s, flags=re.DOTALL).strip()

    # 删除换行符
    s = s.replace('\n', ' ')

    # 删除 emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "]+", flags=re.UNICODE
    )
    s = emoji_pattern.sub("", s)

    # 去除 bullet point 前缀（列点形式）
    bullet_patterns = [
        r'^\s*[-*•●◦▪⦿‣⁃–—]+\s+',  # 各种符号开头的项目符号
        r'^\s*\d+[\.\)]\s+',  # 数字列表：1. 2) 3.
        r'^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*',  # 中文数字序号
    ]
    for pat in bullet_patterns:
        s = re.sub(pat, '', s)

    # 压缩连续空格
    s = re.sub(r'[ \t\u3000]+', ' ', s).strip()
    return s


import torch

def summary_knowledge(strategy_type, knowledge, history, model, tokenizer, max_new_tokens, temperature, device, model_name="llama2"):
    # 处理 knowledge 格式（保持不变）
    try:
        knowledge = "\n".join([f"Post: {t[0]}| : Response: {t[1]}" for i, t in enumerate(knowledge)])
    except:
        knowledge = "\n".join(knowledge)

    if "information" in strategy_type:
        encoded = tokenizer([knowledge], max_length=3000, truncation=True)
        knowledge_ids = encoded.input_ids[0]
        knowledge = tokenizer.decode(
            knowledge_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False
        )

        # ========== prompt 内容完全不变 ==========
        prompt = f'''You are an information-focused knowledge summarizer.
                Your goal: For multiple one-turn dialogues ("post | response" pairs), generate a final summary that captures the essential informational knowledge from all pairs.
                
                Rules:
                1. The summary must be a single continuous paragraph of less than 200 words.
                2. The content must be strictly derived only from the knowledge dialogues.
                3. The summary must not include analysis, explanations, or meta statements.
                4. The summary must not use bullet points, numbering, or formatting symbols.
                
                Process (internal, do not output steps):
                - Identify key topics and keywords from each knowledge dialogue.
                - Write a short summary for each knowledge dialogue capturing its main knowledge.
                - Integrate all individual summaries into one final concise summary retaining important keywords and information.
                
                Example:
                Input:
                Post: I feel nervous about making new friends in a new class. Should I try or just stay alone? | Response: You can first define what friendship means to you, such as emotional support or casual interaction. Then consider whether these needs must be fulfilled within your current environment or through other channels like online platforms. Evaluate the energy, time, and emotional cost of making friends, as well as the potential impact on your studies. If making friends brings positive energy, you can try gradually; otherwise, staying in a comfortable state is also a valid choice.
                Post: I'm scared to speak up in group meetings because I think people will judge me. | Response: Start by observing the group dynamics and identifying one low-risk moment to contribute, such as agreeing with someone's point or asking a clarifying question. Remind yourself that most people are focused on the content rather than judging the speaker. You can also prepare a short comment in advance to reduce anxiety.
                Output:
                The final summarized knowledge: To manage social anxiety in group settings, one can start by clarifying personal social needs and evaluating whether they require in-person fulfillment or alternative channels. Factors such as energy, time, and emotional cost should be considered, with gradual engagement recommended if the experience brings positive energy. For fear of judgment in meetings, observing group dynamics first, choosing low-risk moments to contribute, reminding oneself that listeners focus on content rather than judging the speaker, and preparing comments in advance can help reduce anxiety.
                
                Now generate a summary for the following one-turn dialogues without any explanations within 200 words:
                Input: 
                {knowledge}
                Output:
                The final summarized knowledge:'''

        # ========== 根据 model_name 调整输入格式 ==========
        if "qwen" in model_name.lower():
            messages = [
                {"role": "system", "content": "You are an information-focused knowledge summarizer."},
                {"role": "user", "content": prompt}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
            input_ids = model_inputs.input_ids
        else:
            # Llama2 原始格式
            input_ids = tokenizer([prompt]).input_ids
            input_ids = torch.tensor(input_ids).long().to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        early_stopping=True,
    )

    output_ids = output_ids[0][len(input_ids[0]):]
    knowledge = tokenizer.decode(output_ids, skip_special_tokens=True,
                                          spaces_between_special_tokens=False)
    knowledge = extract_summary(knowledge)
    prefix = "The final summarized knowledge:"
    if knowledge.startswith(prefix):
        knowledge = knowledge[len(prefix):].strip()
    return knowledge


def generate_context_knowledge(history, model, tokenizer, max_new_tokens, temperature, device, model_name="llama2"):
    # 处理 history 格式（保持不变）
    try:
        user_input = "\n".join([f"{t['role']}: {t['content']}" for t in history])
    except (KeyError, TypeError, AttributeError):
        user_input = " ".join(history)

    # ========== prompt 内容完全不变 ==========
    prompt = f'''You are a context-focused knowledge summarizer.
    Your goal: For the dialogue history, generate a final summary that captures the dialogue topic and emotion the seeker shows, meanwhile, figure out the important context-related information needed for future supporter turns.

    Rules:
    1. The summary must be a continuous text paragraph of less than 200 words.
    2. The content must be derived only from the dialogue history.
    3. The summary must not include any analysis, explanations, or list formatting.
    4. The summary should focus primarily on the last dialogue turn or most recent exchange.

    Example:
    Dialogue history:
    Seeker: I've been fighting with my partner every day for the past week. Last night they said they can't do this anymore.
    Supporter: That sounds painful. What have the fights been about?
    Seeker: Mostly money. I lost my part-time job last month. But I think the real issue is I've been shutting down and not talking. Now I'm scared they're going to leave me, but I don't know how to start the conversation.
    Output:
    The final summary: The seeker reveals that daily fights with the partner stem from the seeker's job loss a month ago, but the deeper issue is the seeker's pattern of shutting down emotionally. The partner recently expressed inability to continue. The seeker currently feels scared of abandonment and remains stuck on how to initiate a conversation, with no resolution yet.

    Now generate a summary for the following dialogue without any explanations within 200 words:
    Input:
    Dialogue history:
    {user_input}
    Output:
    The final summary:'''

    # ========== 根据 model_name 调整输入格式 ==========
    if "qwen" in model_name.lower():
        messages = [
            {"role": "system", "content": "You are a context-focused knowledge summarizer."},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
        input_ids = model_inputs.input_ids
    else:
        # Llama2 原始格式
        input_ids = tokenizer([prompt]).input_ids
        input_ids = torch.tensor(input_ids).long().to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        early_stopping=True,
    )

    output_ids = output_ids[0][len(input_ids[0]):]
    knowledge = tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
    return knowledge

def clean_text(text):
    """
    清洗文本：
    - 若输入为 str：清洗换行符并返回。
    - 若输入为 list/tuple（支持多层嵌套）：递归清洗其中所有 str。
    - 若为其他类型：原样返回（可改为抛错）。
    """

    def _clean_single(t: str):
        return t.replace('\n', ' ').replace('\r', ' ')

    def _recursive_clean(obj):
        # 字符串 → 清洗
        if isinstance(obj, str):
            return _clean_single(obj)

        # list/tuple → 递归进入
        if isinstance(obj, (list, tuple)):
            return type(obj)(_recursive_clean(item) for item in obj)

        # 其他类型 → 原样返回
        return obj

    return _recursive_clean(text)

def save_rl_mtric(dataset, filename, epoch, SR, mode='train'):
    PATH = TMP_DIR[dataset] + '/eval_result/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/eval_result/'):
        os.makedirs(TMP_DIR[dataset] + '/eval_result/')
    if mode == 'train':
        with open(PATH, 'a') as f:
            f.write('===========Train===============\n')
            f.write('Starting {} user epochs\n'.format(epoch))
            f.write('training SR: {}\n'.format(SR[0]))
            f.write('training Avg@T: {}\n'.format(SR[1]))
            f.write('training Rewards: {}\n'.format(SR[2]))
            f.write('================================\n')
            # f.write('1000 loss: {}\n'.format(loss_1000))
    elif mode == 'test':
        with open(PATH, 'a') as f:
            f.write('===========Test===============\n')
            f.write('Testing {} user tuples\n'.format(epoch))
            f.write('Testing SR: {}\n'.format(SR[0]))
            f.write('Testing Avg@T: {}\n'.format(SR[1]))
            f.write('Testing Rewards: {}\n'.format(SR[2]))
            f.write('================================\n')