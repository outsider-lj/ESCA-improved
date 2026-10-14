import argparse

# import numpy as np
# from torch.optim import AdamW
from transformers import AutoConfig, LlamaConfig,LlamaTokenizer, RobertaTokenizer, BertConfig, RobertaConfig,AutoModelForCausalLM,AutoTokenizer
from sentence_transformers import SentenceTransformer
import glob
import logging
import os
import random
from pytorch_transformers import WarmupLinearSchedule
import torch
import utils
import data_reader
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from utils import *
from agent_corrected_ppo import MDDP,PG
from prompt import ESConvAct
from sklearn.metrics import f1_score, precision_score, recall_score
from functools import partial
from utils import generate_context_knowledge,summary_knowledge,clean_text
from analysis import save_feature_importance_results,save_attention_results,plot_feature_importance,plot_attention_distribution
# from knowledge.preprocess_add_knowledge import search_faiss_and_fetch_text,get_strategy_category
import re
# from metric import NLGEval
# python sft_dp.py --gpu="0 1" --do_train --overwrite_output_dir --per_gpu_train_batch_size=8 --per_gpu_eval_batch_size=8
PROJECT_DIR="/projects/scc/UGOE/UXEI/UMIN/scc_umin_ag_xiaoming_fu/umin_kurs_datascismartcity_s26/dir.project/lj/LLM"
# EOS/BOS 不再硬编码，统一从 generation_tokenizer 获取
# EOS_TOKEN_ID=2
# BOS_TOKEN_ID=1
tok = {'llama2': LlamaTokenizer,  'roberta': RobertaTokenizer, 'qwen3': AutoTokenizer,'qwen2.5': AutoTokenizer}
cfg = {'llama2': LlamaConfig, 'roberta': RobertaConfig, 'qwen3': AutoConfig,'qwen2.5': AutoConfig}
ACT = sorted(list(ESConvAct.keys()))
from prompt import llama2_prompt

def generate_response( args,model,tokenizer, messages):
    prompt = llama2_prompt(messages,'supporter')
    # print(prompt,flush=True)
    input_ids = tokenizer([prompt]).input_ids
        # print(len(input_ids[0]))
    max_new_tokens =args.max_new_tokens
    output_ids = model.generate(
            torch.as_tensor(input_ids).to(args.pg_device),
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            early_stopping=True
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    # output =tokenizer.decode(output_ids, skip_special_tokens=True,
    #                                           spaces_between_special_tokens=False)
    # 保持 batch 维度 (1, seq_len)，与 generate_from_embedding 返回格式一致
    return output_ids.unsqueeze(0)
def generate_from_embedding(args, model, inputs_embeds: torch.Tensor):
    """
    使用 softprompt (inputs_embeds) 引导模型生成，
    后续生成直接用 HuggingFace 的 generate 方法。
    """
    # device = inputs_embeds.device
    allowed_keys = {
        'max_length', 'min_length', 'temperature', 'top_k', 'top_p', 'num_beams',
        'do_sample', 'repetition_penalty', 'length_penalty', 'bad_words_ids',
        'no_repeat_ngram_size', 'early_stopping',
        'use_cache', 'bos_token_id', 'eos_token_id'
        # 'decoder_input_ids','encoder_outputs', 'past_key_values', 'attention_mask',
    }
    clean_kwargs = {}
    for key in allowed_keys:
        if hasattr(args, key) and getattr(args, key) is not None:
            clean_kwargs[key] = getattr(args, key)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=args.max_new_tokens,  # 必需项
            do_sample=True,  # 默认采样
            use_cache=True,
            **clean_kwargs
        )

    # 注意：generated_ids 包含 soft prompt 之后生成的 token
    # 如果你只想要新生成的部分，可以切掉 inputs_embeds.shape[1]
    if generated_ids.shape[1] > inputs_embeds.shape[1]:
        new_tokens = generated_ids[:, inputs_embeds.shape[1]:]
    else:
        new_tokens = generated_ids

    return new_tokens


class DataFrame(Dataset):
    def __init__(self, data, args):
        self.source_ids = data['source_ids']
        self.emotion_intensity = data['emotion']
        self.trust_scores = data['trust']
        self.stage_ids = data['stage']
        self.behavior_ids = data['behavior']
        self.strategy_ids = data['strategy_ids']

        # self.knowledge_ids = data['retrieved_knowledge']
        self.response_ids = data['response_ids']
        self.gen_source_ids=data['gen_source_ids']
        # self.instruction_ids=data['instruction_ids']
        self.max_len = args.max_seq_length

    def __getitem__(self, index):
        return self.source_ids[index][:self.max_len], self.strategy_ids[index], self.emotion_intensity[index], \
               self.trust_scores[index], self.stage_ids[index], self.behavior_ids[index], \
                self.response_ids[index],self.gen_source_ids[index],

    def __len__(self):
        return len(self.source_ids)


def collate_fn(data, eos_token_id, pad_token_id=None):
    """
    collate_fn 参数说明：
    - eos_token_id: generation tokenizer 的 EOS token ID，用于 labels/response_ids 中作为序列结束标记
    - pad_token_id: generation tokenizer 的 PAD token ID，用于 gen_input_ids 的 padding。
      注意：不能用 eos_token_id 做 padding，因为 gen_input_ids 中每个 turn 末尾有真实的 EOS token，
      用 EOS 做 padding 会导致 gen_attention_mask 把合法 EOS 也 mask 掉。
    """
    source_ids, strategy_ids,emotion_intensity,trust_ids,stage_ids,behavior_ids,response_ids,gen_source_ids= zip(*data)

    emotion = torch.tensor(emotion_intensity).long()
    trust = torch.tensor(trust_ids).long()
    stage = torch.tensor(stage_ids).long()
    behavior = torch.tensor(behavior_ids).long()
    input_ids = [torch.tensor(source_id).long() for source_id in source_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)#PAD_TOKRN_ID in roberta
    attention_mask = input_ids.ne(1)
    labels = [torch.tensor(response_id[1:]).long() for response_id in response_ids]
    labels = pad_sequence(labels, batch_first=True, padding_value=eos_token_id)
    response_ids = [torch.tensor(response_id[:-1]).long() for response_id in response_ids]
    response_ids = pad_sequence(response_ids, batch_first=True, padding_value=eos_token_id)
    strategy_ids = torch.tensor(strategy_ids).long()
    gen_input_ids = [torch.tensor(gen_source_id).long() for gen_source_id in gen_source_ids]
    # 关键修复：用 pad_token_id 而非 eos_token_id 做 padding
    # eos_token_id 在 gen_source_id 中是每个 turn 的合法结束标记，不能和 padding 混淆
    gen_pad_id = pad_token_id if pad_token_id is not None else eos_token_id
    gen_input_ids = pad_sequence(gen_input_ids, batch_first=True, padding_value=gen_pad_id)
    gen_attention_mask = gen_input_ids.ne(gen_pad_id)

    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'strategy_ids': strategy_ids,
            'emotion': emotion,
            'trust': trust,
            'stage': stage,
            'behavior': behavior,
            # 'knowledge_ids': knowledge_ids,
            # 'instruction': instruction_ids,
            'response_ids': response_ids,
            'labels':labels,
            'gen_input_ids': gen_input_ids,
            'gen_attention_mask':gen_attention_mask,
            # 'instruction_ids':instruction_ids
            }


def test(args, dialogue_planner,prompt_generator, tokenizer,generation_model,generation_tokenizer, embed_model,save_output=False):
    test_dataset = data_reader.load_and_cache_examples(args, tokenizer, generation_tokenizer, evaluate=True)
    # eos_token_id = getattr(generation_tokenizer, 'eos_token_id', 2)
    if args.generation_model_name == "llama2":
        bos_id = generation_tokenizer.bos_token_id
        eos_id = generation_tokenizer.eos_token_id
    elif args.generation_model_name == "qwen3" or args.generation_model_name == "qwen2.5":
        bos_id = generation_tokenizer.convert_tokens_to_ids("<|im_start|>")
        eos_id = generation_tokenizer.convert_tokens_to_ids("<|im_end|>")
    # 获取 pad_token_id，用于 gen_input_ids 的 padding（不能用 eos，会与真实 EOS 冲突）
    # pad_id = generation_tokenizer.pad_token_id
    # if pad_id is None:
    #     pad_id = generation_tokenizer.eos_token_id  # fallback，但会有 attention mask 问题
    #     logging.warning("generation_tokenizer.pad_token_id is None, falling back to eos_token_id for padding. "
    #                    "This may cause attention mask issues for short conversations.")
    # logging.info(f"[TEST] eos_id={eos_id}, bos_id={bos_id}, pad_id={pad_id}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, len(args.device_id))

    test_dataloader = DataLoader(DataFrame(test_dataset, args), batch_size=args.eval_batch_size, shuffle=False,
                                 collate_fn=partial(collate_fn, eos_token_id=eos_id, pad_token_id=eos_id))

    # Eval!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(test_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    count = 0
    preds = []
    # response_preds=[]
    targets = []
    scores = []
    sources = []
    responses = []
    golden_responses=[]
    knowledges=[]
    state_attention_results = []
    feature_importance_results = []
    dialogue_planner_to_eval = dialogue_planner.module if hasattr(dialogue_planner, 'module') else dialogue_planner
    dialogue_planner.eval()
    if args.PG==True:
        prompt_generator_to_eval = prompt_generator.module if hasattr(prompt_generator, 'module') else prompt_generator
        prompt_generator.eval()
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        # dialogue planning process

        gen_attention_mask = batch['gen_attention_mask'].to(args.pg_device)
        gen_input_ids = batch['gen_input_ids'].to(args.pg_device)
        conversation = generation_tokenizer.decode(gen_input_ids[0], skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
        segments = re.findall(r'(Supporter:|Seeker:)(.*?)(?=Supporter:|Seeker:|$)', conversation, re.DOTALL)

        # 合并角色和文本，去除前后空格
        conversation = [f"{speaker.strip()} {content.strip()}" for speaker, content in segments]
        conversation_dict = [ {"role": speaker.strip(), "content": content.strip() } for speaker, content in segments]

        # 合并分隔符和内容
        # conversation = [conversation[i] + conversation[i + 1] for i in range(1, len(conversation), 2)]
        # print(conversation,flush=True)
        seq=conversation[-1].strip().lower()
        if seq.startswith("supporter:"):
            print("-------continue-------")
            continue
        sources.extend([
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in batch['input_ids']
        ])
        inputs = {'input_ids': batch['input_ids'].to(args.dp_device),
                      'attention_mask': batch['attention_mask'].to(args.dp_device),
                      'emotion': batch['emotion'].to(args.dp_device),
                      'trust': batch['trust'].to(args.dp_device),
                      'stage': batch['stage'].to(args.dp_device),
                      'behavior': batch['behavior'].to(args.dp_device),
                 "train_mode":"test",
                    }
        with torch.no_grad():
            pred,states,state_attention,feature_importance= dialogue_planner_to_eval(**inputs)
        scores.extend([p[0] for p in pred.cpu().tolist()])
        pre_strategy_id=pred.argmax(dim=-1).cpu().tolist()
        pre_strategy=ACT[pre_strategy_id[0]]
        preds.extend(pre_strategy_id)
        targets.extend(batch['strategy_ids'].tolist())
        state_attention_results.append(state_attention)
        feature_importance_results.append(feature_importance)
        precision = precision_score(targets, preds, average='macro', zero_division=0)
        recall = recall_score(targets, preds, average='macro', zero_division=0)
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
        auto_scores = [precision, recall, f1]
        # generation process
        context=" ".join(conversation[-4:])
        if args.RAG == True:
            category = get_strategy_category(pre_strategy)
            if category == "information-related":
                k = search_faiss_and_fetch_text(embed_model, context, args.psyqa_index_path,
                                                 args.psyqa_db_path,top_k=2)
                k = clean_text(k)
                k = summary_knowledge(category, k, context, generation_model, generation_tokenizer, max_new_tokens=200,
                                      temperature=0.7, device=args.pg_device,model_name=args.generation_model_name)

                retrieved_knowledge = "The "+category + " knowledge: " + k
                retrieved_knowledge=data_reader.clean_knowledge(retrieved_knowledge)
            elif category == "context-related":
                k = generate_context_knowledge(context, generation_model,generation_tokenizer, max_new_tokens=200,
                                      temperature=0.7, device=args.pg_device,model_name=args.generation_model_name)
                k = clean_text(k)
                retrieved_knowledge ="The "+ category + " knowledge: " + k
                retrieved_knowledge =data_reader.clean_knowledge(retrieved_knowledge)
            else:
                print("Something wrong with category",flush=True)
        else:
            retrieved_knowledge = " "
        if args.RAG == True:
            knowledge_ids = ([bos_id] if bos_id is not None else [])+generation_tokenizer.encode(retrieved_knowledge,add_special_tokens=False)[:args.max_seq_length] + ([eos_id] if eos_id is not None else [])
            knowledge_ids = torch.tensor([knowledge_ids]).long().to(args.pg_device)
            with torch.no_grad():
                knowledge_embeddings = generation_model.model.embed_tokens(knowledge_ids)
        else:
            knowledge_embeddings=None
        knowledges.append(retrieved_knowledge)
        instruction = ESConvAct[pre_strategy]
        if args.PG==True:
            with torch.no_grad():
                context_embeddings = generation_model.model.embed_tokens(gen_input_ids)  # [1, seq_len, hidden_size]

                instruction_ids = ([bos_id] if bos_id is not None else [])+generation_tokenizer.encode(instruction,add_special_tokens=False) + ([eos_id] if eos_id is not None else [])
                instruction_ids = torch.tensor([instruction_ids]).long().to(args.pg_device)
                instruction_embeddings = generation_model.model.embed_tokens(instruction_ids)
                embedding=prompt_generator_to_eval(context_embeddings= context_embeddings,
                                         attention_mask=gen_attention_mask,
                          knowledge_embeddings=knowledge_embeddings,
                                                   instruction_embeddings=instruction_embeddings,
                                                   train_mode='test'
                          )
                # bsz=embedding.size(0)
                First_tokens=generation_tokenizer.encode("supporter:",add_special_tokens=False)
                First_embedding=generation_model.model.embed_tokens(torch.tensor([First_tokens]).long().to(args.pg_device))
                embedding=torch.cat([embedding,First_embedding],dim=1)
                response_pred=generate_from_embedding(args,generation_model,embedding)
        else:
            if retrieved_knowledge is not None:
                knowledge_prompt = f"\nNote: Here is some retrieved knowledge that might be helpful:\n{retrieved_knowledge}\n\n"
            else:
                knowledge_prompt =" "
            # context = " ".join([f"{t['speaker']}: {t['text']}" for t in conversation])
            context = " ".join(conversation)
            messages = [
                    {"role": "system",
                     "content": f"Now enter the role-playing mode. In the following conversation, you will play as a supporter in a emotional support conversation with a Seeker."},
                    {"role": "usr",
                     "content": f"{instruction} {knowledge_prompt} The dialogue context is: {context} Please reply with only one short and succinct sentence."},
                ]
            # messages = [
            #     {"role": "system",
            #      "content": f"Now enter the role-playing mode. In the following conversation, you will play as a supporter in a emotional support conversation with a Seeker.{knowledge_prompt}"},
            #     {"role": "Seeker",
            #      "content": f"{instruction} Please reply with only one short and succinct sentence.  Are you ready to play the game?"},
            #     {"role": "Supporter", "content": "Yes, I'm ready to play the game!"}
            # ]
            # messages.extend(conversation_dict)
            response_pred=generate_response(args,generation_model,generation_tokenizer,messages)
        decoded = generation_tokenizer.decode(
            response_pred[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # 去除开头的 \n、空白和 "Supporter:" / "supporter:" 前缀
        print(decoded,flush=True)
        decoded = decoded.lstrip("\n ").removeprefix("Supporter:").removeprefix("supporter:").lstrip()
        responses.extend([decoded])
        golden_responses.extend([
            generation_tokenizer.decode(
                r, skip_special_tokens=True, clean_up_tokenization_spaces=False).lstrip("\n ").removeprefix("Supporter:").removeprefix("supporter:").lstrip()
            for r in batch['response_ids']
        ])
        dist=compute_distinct2(responses)
        bleu=compute_bleu4(responses,golden_responses)
        auto_scores=auto_scores+[dist,bleu]
            # metric_res, auto_scores = metric.compute_metrics([golden_responses],response_preds )
        if save_output:
            REC_PATH ='./results/test_result/' + args.output_dir+args.generation_model_name + '-42-64.txt'
            if not os.path.isdir('./results/test_result/' + args.output_dir ):
                os.makedirs('./results/test_result/' + args.output_dir)
            with open(REC_PATH, 'w') as result_file:
                for target, pred, response_pred, golden_resp,knowledge, source in zip(targets, preds, responses, golden_responses,knowledges,sources):
                    result_file.write('%s\n\n' % str({'Context': source, 'Pre_Strategy': pred,'True_Strategy': target,'Knowledge':knowledge,'Pre_Response':response_pred,'Golden_Response':golden_resp}))
            if args.state_attention==True:
                save_attention_results(state_attention_results, 'state_attention.json')
                save_feature_importance_results(feature_importance_results, 'feature_importance.json')

                # 生成图表
                plot_attention_distribution(state_attention_results, 'state_attention.png')
                plot_feature_importance(feature_importance_results, 'feature_importance.png')
    logging.info(auto_scores)
    print(auto_scores)
    return auto_scores

def main():
    parser = argparse.ArgumentParser(description="train.py")

    ## Required parameters
    parser.add_argument('--data_name', default='esc', type=str,
                        help="dataset name")
    parser.add_argument('--set_name', default='test', type=str,
                        help="dataset split name")
    parser.add_argument('--model_name', default='roberta', type=str,
                        help="model name")
    parser.add_argument('--generation_model_name', default='qwen2.5', type=str,
                        help="model name")
    parser.add_argument('--model_name_or_path', default=PROJECT_DIR+"/roberta-large", type=str,
                        help="model name")
    parser.add_argument('--generation_model_name_or_path', default=PROJECT_DIR+"/qwen2.5-3b", type=str,#/llama2-7b-chat-hf /qwen3-8b
                        help="model name")
    parser.add_argument("--output_dir", default='test', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", default='./data', type=str,
                        help="The data directory.")
    parser.add_argument("--cache_dir", default='./storage_fast/plm', type=str,
                        help="The cache directory.")
    parser.add_argument("--embed_model_path", type=str,
                        default=PROJECT_DIR+'/all-MiniLM-L6-v2')
    ## Other parameters
    parser.add_argument("--do_train", default=False,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True,
                        help="Whether to run eval.")
    parser.add_argument("--dp_learning_rate", default=1e-6, type=float,#5
                        help="The initial learning rate for Adam.")
    parser.add_argument("--pg_learning_rate", default=1e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--overwrite_output_dir', default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', default=True,
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")
    parser.add_argument('--gpu', default="0", type=str,
                        help="Use CUDA on the device.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="DDP requirement.")
    parser.add_argument('--dp_sft_dir', default=PROJECT_DIR+'/Dialogue_Planning_LLM/sft/esc/roberta/Truestate_attention/Truerag/best_checkpoint/dialogue_planner',#'/mnt/vast-standard/home/jli6/u27481/Dialogue_Planning_LLM/tmp/esc/RL-agent/esc-qwen2.5-chat_ai-chat_aidp-epoch-1',#'tmp/esc/RL-agent/esc-llama2-chat_ai-chat_aidp-epoch-1',#
                        # ../pretrain/outputs/best_pretrain.pt #/mnt/vast-standard/home/jli6/u27481/Dialogue_Planning_LLM/tmp/esc/RL-agent/esc-llama2-chat_ai-chat_aidp-epoch-1#PROJECT_DIR+'/Dialogue_Planning_LLM/sft/esc/roberta/Truestate_attention/Truerag/best_checkpoint/dialogue_planner',#
                        type=str, help="Pretrain model path.")#写入rl后存入的位置
    parser.add_argument('--pg_sft_dir', default=PROJECT_DIR+'/Dialogue_Planning_LLM/sft/esc/roberta/Truestate_attention/Truerag/best_checkpoint/prompt_generator',#'/mnt/vast-standard/home/jli6/u27481/Dialogue_Planning_LLM/tmp/esc/RL-agent/esc-qwen2.5-chat_ai-chat_aipg-epoch-1',#PROJECT_DIR+'/Dialogue_Planning_LLM/sft/esc/roberta/Truestate_attention/Truerag/best_checkpoint/prompt_generator',
                        type=str, help="Pretrain model path.")

    # parser.add_argument("--chat_ai_model", type=str,
    #                     default="llama-3.3-70b-instruct")  # qwen2.5-72b-instruct #meta-llama-3.1-70b-instruct #deepseek-r1-distill-llama-70b
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.03)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument('--eos_token_id', type=int, default=2)

    parser.add_argument("--psyqa_index_path", type=str, default="./knowledge/psyqa_knowledge_faiss.index",
                        help="Path to FAISS index for PsyQA")
    parser.add_argument("--psyqa_db_path", type=str, default="./knowledge/psyqa_knowledge.db",
                        help="CSV file for PsyQA knowledge")

    parser.add_argument("--state_attention", default=True, type=bool,
                        help="if add four state aspects")
    parser.add_argument("--PG", default=False, type=bool,
                        help="if add PG to LLM")
    parser.add_argument("--RAG", type=bool, default=True)
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.data_name, args.model_name,str(args.state_attention)+"state_attention",str(args.RAG)+"rag")
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(level=logging.DEBUG, filename=args.output_dir + '/log.txt', filemode='a')

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    # ===== [新增] 双 GPU 设备分配 =====
    # dialog_planner (MDDP) -> GPU 0
    # prompt_generator (PG) -> GPU 1
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        args.dp_device = torch.device('cuda:0')  # strategy_planner 使用 GPU 0
        args.pg_device = torch.device('cuda:1')  # prompt_generator 使用 GPU 1
        args.device = args.dp_device  # 保留兼容性，默认指向 GPU 0
        print(f"[GPU分配] dialog_planner -> cuda:0, prompt_generator -> cuda:1")
    elif torch.cuda.is_available():
        # 只有1个 GPU 时退回到单卡模式
        args.dp_device = torch.device('cuda:0')
        args.pg_device = torch.device('cuda:0')
        args.device = torch.device('cuda:0')
        print("[GPU分配] 只检测到1个GPU，两个模型共享 cuda:0")
    else:
        args.dp_device = torch.device('cpu')
        args.pg_device = torch.device('cpu')
        args.device = torch.device('cpu')
        print("[GPU分配] 未检测到GPU，使用 CPU")
    # ===== [新增结束] =====

    config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case,
                                                     cache_dir=args.cache_dir)
    generation_config = cfg[args.generation_model_name].from_pretrained(args.generation_model_name_or_path, cache_dir=args.cache_dir, trust_remote_code=True)
    generation_tokenizer = tok[args.generation_model_name].from_pretrained(args.generation_model_name_or_path,
                                                         do_lower_case=args.do_lower_case, trust_remote_code=True)
    dialogue_planner = MDDP(args, config, tokenizer)
    dialogue_planner.to(args.dp_device)
    generation_model = AutoModelForCausalLM.from_pretrained(args.generation_model_name_or_path, trust_remote_code=True)
    generation_model.to(args.pg_device)
    generation_model.eval()
    if args.generation_model_name=="llama2":
        args.bos_token_id=generation_tokenizer.bos_token_id
        args.eos_token_id=generation_tokenizer.eos_token_id
    elif args.generation_model_name=="qwen3" or args.generation_model_name=="qwen2.5":
        args.bos_token_id = generation_tokenizer.convert_tokens_to_ids("<|im_start|>")
        args.eos_token_id = generation_tokenizer.convert_tokens_to_ids("<|im_end|>")
    for param in generation_model.parameters():
        param.requires_grad = False
    if args.PG==True:
        prompt_generator = PG(args, generation_config, generation_tokenizer)
        prompt_generator.to(args.pg_device)
    else:
        prompt_generator=None


    logging.info("Evaluation parameters %s", args)
    if args.dp_sft_dir is not None:
        print('Staring loading policy model from {}'.format(args.dp_sft_dir))
        dialogue_planner.load_model(data_name=args.data_name, filename=args.dp_sft_dir)
        dialogue_planner.to(args.dp_device)
        print(next(dialogue_planner.parameters()).device,flush=True)
    if args.pg_sft_dir is not None and args.PG==True:
        print('Staring loading policy model from {}'.format(args.pg_sft_dir))
        prompt_generator.load_model(data_name=args.data_name, filename=args.pg_sft_dir)
        prompt_generator.to(args.pg_device)
        print(next(dialogue_planner.parameters()).device,flush=True)

    # dataset = load_dataset(args.data_name)
    # if hasattr(dialogue_planner, 'module'):
    #     dialogue_planner.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
    # else:
    #     dialogue_planner.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
    # if hasattr(prompt_generator, 'module'):
    #     prompt_generator.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
    # else:
    #     prompt_generator.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
    if args.RAG == True:
        embed_model = SentenceTransformer(args.embed_model_path)
    else:
        embed_model=None
    args.set_name = 'test'
    test(args, dialogue_planner,prompt_generator, tokenizer,generation_model,generation_tokenizer,embed_model, save_output=True)


if __name__ == "__main__":
    main()