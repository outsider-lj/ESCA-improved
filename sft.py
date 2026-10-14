import argparse
import numpy as np
from torch.optim import AdamW
from transformers import AutoConfig, LlamaConfig, LlamaTokenizer, RobertaTokenizer, BertConfig, RobertaConfig, \
    AutoModelForCausalLM, AutoTokenizer
import glob
import logging
import random
from pytorch_transformers import WarmupLinearSchedule
import torch
import utils
import data_reader
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from utils import compute_bleu4,compute_distinct2
from agent_corrected_ppo import MDDP,PG
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.metrics import f1_score, precision_score, recall_score

PROJECT_DIR="/projects/scc/UGOE/UXEI/UMIN/scc_umin_ag_xiaoming_fu/umin_kurs_datascismartcity_s26/dir.project/lj/LLM"
tok = {'llama2': LlamaTokenizer,  'roberta': RobertaTokenizer, 'qwen3': AutoTokenizer,'qwen2.5': AutoTokenizer}
cfg = {'llama2': LlamaConfig, 'roberta': RobertaConfig, 'qwen3': AutoConfig,'qwen2.5': AutoConfig}
import os
import torch.distributed as dist

def safe_save(obj, path):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)   # 原子替换

def setup_ddp(args):
    """
    Standard DDP init for torchrun
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")

        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)

        return True
    else:
        args.rank = 0
        args.local_rank = -1
        args.world_size = 1
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False


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
        self.knowledge_ids = data['retrieved_knowledge']
        self.response_ids = data['response_ids']
        self.gen_source_ids=data['gen_source_ids']
        self.instruction_ids=data['instruction_ids']
        self.max_len = args.max_seq_length

    def __getitem__(self, index):
        return self.source_ids[index][:self.max_len], self.strategy_ids[index], self.emotion_intensity[index], \
               self.trust_scores[index], self.stage_ids[index], self.behavior_ids[index], \
               self.knowledge_ids[index], self.response_ids[index],self.gen_source_ids[index],self.instruction_ids[index]

    def __len__(self):
        return len(self.source_ids)


from functools import partial

def collate_fn(data, eos_token_id,pad_token_id):
    source_ids, strategy_ids,emotion_intensity,trust_ids,stage_ids,behavior_ids,knowledge_ids,response_ids,gen_source_ids,instruction_ids = zip(*data)
    input_ids = [torch.tensor(source_id).long() for source_id in source_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    attention_mask = input_ids.ne(1)
    emotion = torch.tensor(emotion_intensity).long()
    trust = torch.tensor(trust_ids).long()
    stage = torch.tensor(stage_ids).long()
    behavior = torch.tensor(behavior_ids).long()
    labels = [torch.tensor(response_id[1:]).long() for response_id in response_ids]
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    response_ids = [torch.tensor(response_id[:-1]).long() for response_id in response_ids]
    response_ids = pad_sequence(response_ids, batch_first=True, padding_value=pad_token_id)
    knowledge_ids = [torch.tensor(knowledge_id).long() for knowledge_id in knowledge_ids]
    knowledge_ids = pad_sequence(knowledge_ids, batch_first=True, padding_value=pad_token_id)
    knowledge_attention_mask = knowledge_ids.eq(pad_token_id)
    strategy_ids = torch.tensor(strategy_ids).long()
    gen_input_ids = [torch.tensor(gen_source_id).long() for gen_source_id in gen_source_ids]
    gen_input_ids = pad_sequence(gen_input_ids, batch_first=True, padding_value=pad_token_id)
    gen_attention_mask = gen_input_ids.ne(pad_token_id)
    instruction_ids = [torch.tensor(instruction_id).long() for instruction_id in instruction_ids]
    instruction_ids = pad_sequence(instruction_ids, batch_first=True, padding_value=pad_token_id)
    instruction_attention_mask=instruction_ids.eq(pad_token_id)
    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'strategy_ids': strategy_ids,
            'emotion': emotion,
            'trust': trust,
            'stage': stage,
            'behavior': behavior,
            'knowledge_ids': knowledge_ids,
            'knowledge_mask':knowledge_attention_mask,
            # 'instruction': instruction_ids,
            'response_ids': response_ids,
            'labels':labels,
            'gen_input_ids': gen_input_ids,
            'gen_attention_mask':gen_attention_mask,
            'instruction_ids':instruction_ids,
            'instruction_mask':instruction_attention_mask
            }
def get_model(model):
    return model.module if hasattr(model, "module") else model

def train(args, train_dataset, dialogue_planner,prompt_generator, tokenizer,generation_model,generation_tokenizer, eos_token_id=None,pad_token_id=None):
    # Initialize distributed training
    # torch.distributed.init_process_group(backend="nccl")
    # local_rank = torch.distributed.get_rank()
    is_ddp = args.world_size > 1

    local_rank = args.local_rank
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    else:
        tb_writer = None

    # 使用 generation_tokenizer 的 eos_token_id
    # if eos_token_id is None:
    #     eos_token_id = getattr(generation_tokenizer, 'eos_token_id', 2)

    # Initialize DistributedSampler if running in distributed mode
    if is_ddp:
        sampler = DistributedSampler(
            DataFrame(train_dataset, args),
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True
        )
    else:
        sampler = None

    print("Local rank",local_rank,flush=True)
    # Create DataLoader with DistributedSampler
    train_dataloader = DataLoader(
        DataFrame(train_dataset, args),
        sampler=sampler,
        batch_size=args.per_gpu_train_batch_size,
        collate_fn=partial(collate_fn, eos_token_id=eos_token_id,pad_token_id=pad_token_id),
        shuffle=(sampler is None),
        # num_workers=args.dataloader_num_workers,
    )

    args.train_batch_size = (
            args.per_gpu_train_batch_size
            * args.world_size
            * args.gradient_accumulation_steps
    )

    # train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.local_rank != -1 else None
    # train_dataloader = DataLoader(DataFrame(train_dataset, args), batch_size=args.train_batch_size, sampler=train_sampler,
    #                               collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    if "sft_dp" in args.train_process:

        if is_ddp:
            dialogue_planner = DDP(
                dialogue_planner,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
            )
        dp = get_model(dialogue_planner)
        dp.zero_grad()
        dialogue_planner_scheduler = WarmupLinearSchedule(dp.optimizer, warmup_steps=args.warmup_steps,
                                                              t_total=t_total)
    if "sft_pg" in args.train_process:
        if is_ddp:
            prompt_generator = DDP(
                    prompt_generator,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    find_unused_parameters=True
                )
        pg = get_model(prompt_generator)
        pg.zero_grad()
        generation_model.eval()
        prompt_generator_scheduler = WarmupLinearSchedule(pg.optimizer, warmup_steps=args.warmup_steps,
                                                          t_total=t_total)
    # Train
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                 args.train_batch_size * args.gradient_accumulation_steps)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    tr_str_loss, logging_str_loss , tr_gen_loss, logging_gen_loss = 0.0, 0.0, 0.0, 0.0

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    utils.set_random_seed(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    global_step = 0
    # total_rouge = evaluate(args, model, tokenizer, save_output=True)
    best_pr = 0
    best_ppl = 50# total_rouge[2]
    # Log metrics

    for e in train_iterator:
        if sampler is not None:
            sampler.set_epoch(e)
        logging.info("training for epoch {} ...".format(e))
        print("training for epoch {} ...".format(e))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            if "sft_dp" in args.train_process:
                strategy_label = batch["strategy_ids"].to(args.device)
                inputs = {'input_ids': batch['input_ids'].to(args.device),
                          # 'gen_input_ids': batch['gen_input_ids'].to(args.device),
                          'attention_mask': batch['attention_mask'].to(args.device),
                          'emotion': batch['emotion'].to(args.device),
                          'trust': batch['trust'].to(args.device),
                          'stage': batch['stage'].to(args.device),
                          'behavior': batch['behavior'].to(args.device),
                          }
                strategy_logits, states = dialogue_planner(**inputs)
                strategy_loss = torch.nn.CrossEntropyLoss()(strategy_logits.view(-1, 8), strategy_label.view(-1))
                # if len(args.device_id) > 1:
                #     strategy_loss = strategy_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    strategy_loss = strategy_loss / args.gradient_accumulation_steps

                strategy_loss.backward()
                tr_str_loss += strategy_loss.item()
                torch.nn.utils.clip_grad_norm_(dp.parameters(), args.max_grad_norm)
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    dialogue_planner_scheduler.step()
                    dp.optimizer.step()
                    dp.zero_grad()
            if "sft_pg" in args.train_process:
                response_ids = batch['response_ids'].to(args.device)
                gen_input_ids = batch['gen_input_ids'].to(args.device)
                instruction_ids = batch['instruction_ids'].to(args.device)
                labels = batch['labels'].to(args.device)
                knowledge_attention_mask = batch['knowledge_mask'].to(args.device)
                instruction_attention_mask = batch['instruction_mask'].to(args.device)
                gen_attention_mask = batch['gen_attention_mask'].to(args.device)
                context_embeddings = generation_model.model.embed_tokens(gen_input_ids)  # [1, seq_len, hidden_size]
                instruction_embeddings = generation_model.model.embed_tokens(instruction_ids)
                if args.RAG == True:
                    knowledge_ids = batch['knowledge_ids'].to(args.device)
                    knowledge_embeddings = generation_model.model.embed_tokens(knowledge_ids)
                else:
                    knowledge_embeddings = None
                embeddings=prompt_generator(
                          context_embeddings=context_embeddings,

                          knowledge_embeddings=knowledge_embeddings,
                                              instruction_embeddings=instruction_embeddings,
                    attention_mask=gen_attention_mask,
                    knowledge_mask=knowledge_attention_mask,
                    instruction_mask=instruction_attention_mask
                )
                bsz=embeddings.size(0)
                query_att1= torch.ones(bsz,8).to(args.device).ne(pad_token_id)
                query_att2 = torch.ones(bsz, 64).to(args.device).ne(pad_token_id)
                response_mask=response_ids.ne(pad_token_id)
                final_attention_mask=torch.cat([query_att1,query_att2,gen_attention_mask,response_mask],dim=1)
                response_embeddings=generation_model.model.embed_tokens(response_ids)
                final_embeddings=torch.cat([embeddings,response_embeddings],dim=1)
                # 模型已通过 torch_dtype 加载为 bf16/fp32，直接前向即可
                final_outputs = generation_model(inputs_embeds=final_embeddings,attention_mask=final_attention_mask)
                gen_logits = final_outputs.logits
                input_length=embeddings.size(1)
                gen_logits=gen_logits[:,input_length:]
                gen_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(gen_logits.contiguous().view(-1, gen_logits.size(-1)), labels.view(-1))
                # loss = outputs  # [0]  # model outputs are always tuple in pytorch-transformers (see doc)
                # if len(args.device_id) > 1:
                #     gen_loss = gen_loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    gen_loss = gen_loss / args.gradient_accumulation_steps
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(pg.parameters(), args.max_grad_norm)
                tr_gen_loss += gen_loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    prompt_generator_scheduler.step()
                    pg.optimizer.step()
                    pg.zero_grad()
            global_step += 1
        if "sft_dp" in args.train_process:
            if tb_writer :
                tb_writer.add_scalar('dp_lr', dialogue_planner_scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar('str_loss', (tr_str_loss - logging_str_loss) / (step + 1), global_step)
            print('str_loss: {}'.format((tr_str_loss - logging_str_loss) / (step + 1)))
        if "sft_pg" in args.train_process:
            if tb_writer :
                tb_writer.add_scalar(
                    'pg_lr',
                    prompt_generator_scheduler.get_last_lr()[0],
                    global_step
                )

            # tb_writer.add_scalar('pg_lr', prompt_generator_scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('gen_loss', (tr_gen_loss - logging_gen_loss) / (step + 1), global_step)
            print('gen_loss: {}'.format((tr_gen_loss - logging_gen_loss) / (step + 1)))
        logging_str_loss = tr_str_loss
        logging_gen_loss = tr_gen_loss

        # Log metrics
        results = evaluate(args, dialogue_planner,prompt_generator, tokenizer,generation_model,generation_tokenizer, save_output=False, eos_token_id=eos_token_id,pad_token_id=pad_token_id)

            # Save model checkpoint
        if "sft_dp" in args.train_process:
            if results[0] > best_pr and e>0:
                best_pr = results[0]
                output_dir = os.path.join(args.output_dir, 'best_checkpoint')
                dialogue_planner_output_dir=os.path.join(output_dir, 'dialogue_planner')
                if not os.path.exists(dialogue_planner_output_dir):
                    os.makedirs(dialogue_planner_output_dir)
                if is_ddp:
                    dist.barrier()
                if args.rank == 0:
                    model_to_save = (
                        dialogue_planner.module
                        if hasattr(dialogue_planner, "module")
                        else dialogue_planner
                    )

                    save_path = os.path.join(
                        dialogue_planner_output_dir,
                        "pytorch_model.bin"
                    )
                    safe_save(model_to_save.state_dict(), save_path)
                    safe_save(args, os.path.join(output_dir, "training_args.bin"))
                    logging.info("Saving model checkpoint to %s", output_dir)
        if "sft_pg" in args.train_process or "DPO" in args.train_process:
            if results[0] < best_ppl:
                best_ppl = results[0]
                output_dir = os.path.join(args.output_dir, 'best_checkpoint')
                prompt_generator_output_dir = os.path.join(output_dir, 'prompt_generator-qwen2.5')
                if not os.path.exists(prompt_generator_output_dir):
                    os.makedirs(prompt_generator_output_dir)
                # save dialogue planner
                if is_ddp:
                    dist.barrier()
                if args.rank == 0:
                    model_to_save = (
                        prompt_generator.module
                        if hasattr(prompt_generator, "module")
                        else prompt_generator
                    )

                    save_path = os.path.join(
                        prompt_generator_output_dir,
                        "pytorch_model.bin"
                    )

                    safe_save(model_to_save.state_dict(), save_path)

                    safe_save(args, os.path.join(output_dir, "training_args.bin"))

                    logging.info("Saving model checkpoint to %s", output_dir)

    tb_writer.close()
    return global_step, tr_str_loss / global_step,tr_gen_loss / global_step


def evaluate(args, dialogue_planner,prompt_generator, tokenizer,generation_model,generation_tokenizer, save_output=False, eos_token_id=None,pad_token_id=None):
    eval_dataset = data_reader.load_and_cache_examples(args, tokenizer,generation_tokenizer, evaluate=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, len(args.device_id))
    # 使用 generation_tokenizer 的 eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(generation_tokenizer, 'eos_token_id', 2)

    all_metrics = []
    eval_dataloader = DataLoader(DataFrame(eval_dataset, args), batch_size=args.eval_batch_size, shuffle=False,
                                 collate_fn=partial(collate_fn, eos_token_id=eos_token_id,pad_token_id=pad_token_id))
    # Eval!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    count = 0
    preds = []
    # response_preds=[]
    targets = []
    scores = []
    sources = []
    responses = []
    golden_responses=[]
    auto_scores=[]
    tr_gen_loss=[]
    str_losses=[]
    ratios=[]
    if "sft_dp" in args.train_process:
        dialogue_planner_to_eval = dialogue_planner.module if hasattr(dialogue_planner, 'module') else dialogue_planner
    if "sft_pg" in args.train_process :
        prompt_generator_to_eval = prompt_generator.module if hasattr(prompt_generator, 'module') else prompt_generator
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        sources.extend([
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for g in batch['input_ids']
            ])
        if "sft_dp" in args.train_process:
            dialogue_planner.eval()
            strategy_label=batch["strategy_ids"].to(args.device)
            inputs = {'input_ids': batch['input_ids'].to(args.device),
                          # 'gen_input_ids': batch['gen_input_ids'].to(args.device),
                          'attention_mask': batch['attention_mask'].to(args.device),
                          'emotion': batch['emotion'].to(args.device),
                          'trust': batch['trust'].to(args.device),
                          'stage': batch['stage'].to(args.device),
                          'behavior': batch['behavior'].to(args.device),
                          # 'knowledge_ids': batch['knowledge_ids'].to(args.device),
                        }
            with torch.no_grad():
                pred,states= dialogue_planner_to_eval(**inputs)
                strategy_loss = torch.nn.CrossEntropyLoss()(pred.view(-1, 8), strategy_label.view(-1))
                # 直接用梯度×输入方法
                # ratio = dialogue_planner_to_eval.analyze_feature_importance(state_features, cls_output)
            str_losses.append(strategy_loss.item())
            scores.extend([p[0] for p in pred.cpu().tolist()])
            preds.extend(pred.argmax(dim=-1).cpu().tolist())
            targets.extend(batch['strategy_ids'].tolist())
        if "sft_pg" in args.train_process:
            prompt_generator.eval()
            gen_attention_mask = batch['gen_attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            gen_input_ids = batch['gen_input_ids'].to(args.device)
            instruction_ids = batch['instruction_ids'].to(args.device)
            response_ids = batch['response_ids'].to(args.device)
            knowledge_attention_mask = batch['knowledge_mask'].to(args.device)
            instruction_attention_mask = batch['instruction_mask'].to(args.device)
            context_embeddings = generation_model.model.embed_tokens(gen_input_ids)  # [1, seq_len, hidden_size]
            instruction_embeddings = generation_model.model.embed_tokens(instruction_ids)
            if args.RAG==True:
                knowledge_ids = batch['knowledge_ids'].to(args.device)
                knowledge_embeddings = generation_model.model.embed_tokens(knowledge_ids)
            else:
                knowledge_embeddings=None
            with torch.no_grad():
                embedding=prompt_generator_to_eval(context_embeddings= context_embeddings,
                                         attention_mask=gen_attention_mask,
                          knowledge_embeddings=knowledge_embeddings,
                                                   instruction_embeddings=instruction_embeddings,
                                                   knowledge_mask=knowledge_attention_mask,
                                                   instruction_mask=instruction_attention_mask
                          )
                bsz=embedding.size(0)
                if save_output==False:
                    query_att1 = torch.ones(bsz, 8).to(args.device).ne(pad_token_id)
                    query_att2 = torch.ones(bsz,64).to(args.device).ne(pad_token_id)
                    response_mask = response_ids.ne(pad_token_id)
                    final_attention_mask = torch.cat([query_att1, query_att2, gen_attention_mask, response_mask], dim=1)
                    response_embedding = generation_model.model.embed_tokens(response_ids)
                    final_embedding = torch.cat([embedding, response_embedding], dim=1)
                    final_outputs = generation_model(inputs_embeds=final_embedding,attention_mask=final_attention_mask)
                    gen_logits = final_outputs.logits
                    input_length = embedding.size(1)
                    gen_logits = gen_logits[:, input_length:]
                    gen_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(gen_logits.contiguous().view(-1, gen_logits.size(-1)),
                                                           labels.view(-1))
                    # loss = outputs  # [0]  # model outputs are always tuple in pytorch-transformers (see doc)
                    if len(args.device_id) > 1:
                        gen_loss = gen_loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        gen_loss = gen_loss / args.gradient_accumulation_steps
                    tr_gen_loss.append(gen_loss.item())
                else:
                    First_tokens=generation_tokenizer.encode("supporter:",add_special_tokens=False)
                    First_tokens_tensor= torch.tensor([First_tokens]).long().to(args.device).repeat(bsz,1)
                    First_embedding = generation_model.model.embed_tokens(First_tokens_tensor)
                    embedding = torch.cat([embedding, First_embedding], dim=1)
                    response_preds = generate_from_embedding(args,generation_model, embedding)
                # metric.forward(batch['response_ids'][1:],response_preds)
                # response_preds.extend([resp[0] for resp in response_pred.cpu().tolist()])
            # metric_res, auto_scores = metric.compute_metrics([golden_responses],response_preds )
    if "sft_dp" in args.train_process:
        if save_output:
            with open(os.path.join(args.output_dir,
                                   '{}_{}_{}.score'.format(args.data_name, args.set_name, list(filter(None,
                                                                                                      args.model_name_or_path.split(
                                                                                                          '/'))).pop())),
                      'w') as outfile:
                for target, pred, score, source in zip(targets, preds, scores, sources):
                    outfile.write(
                        "Target Strategy:{}\tPred Strategy:{}\tScore:{}\nSource:{}\n\n".format(target, pred, score,
                                                                                     source))
        str_loss = np.mean(str_losses)
        precision = precision_score(targets, preds, average='macro', zero_division=0)
        recall = recall_score(targets, preds, average='macro', zero_division=0)
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
        auto_scores = [precision, recall, f1,str_loss]
    elif "sft_pg" in args.train_process:
        if save_output:
                responses.extend([
                    generation_tokenizer.decode(
                        r, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for r in response_preds
                ])
                golden_responses.extend([
                    generation_tokenizer.decode(
                        r, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    for r in batch['response_ids']
                ])
                dist = compute_distinct2(responses)
                bleu = compute_bleu4(responses, golden_responses)
                auto_scores = auto_scores + [bleu,dist]
                REC_PATH ='./results/eval_result/' + args.output_dir + '.txt'
                if not os.path.isdir('./results/eval_result/' + args.output_dir ):
                    os.makedirs('./results/eval_result/' + args.output_dir)
                with open(REC_PATH, 'w') as result_file:
                    for target, pred, response_pred, golden_resp,source in zip(targets, preds, responses, golden_responses,sources):
                        result_file.write('%s\n\n' % str({'Context': source, 'Pre_Strategy': pred,'True_Strategy': target,'Pre_Response':response_pred,'Golden_Response':golden_resp}))
        else:
            loss=np.mean(tr_gen_loss)
            ppl=np.exp(loss)
            auto_scores=[ppl]
    logging.info(auto_scores)
    print(auto_scores,flush=True)
    return auto_scores

def main():
    parser = argparse.ArgumentParser(description="train.py")
    ## Required parameters
    parser.add_argument('--data_name', default='esc', type=str,
                        help="dataset name")
    parser.add_argument('--set_name', default='valid', type=str,
                        help="dataset split name")
    parser.add_argument('--model_name', default='roberta', type=str,
                        help="model name")
    parser.add_argument('--generation_model_name', default='qwen2.5', type=str,
                        help="model name")
    parser.add_argument('--model_name_or_path', default=PROJECT_DIR+"/roberta-large", type=str,
                        help="model name")#./../roberta-large
    parser.add_argument('--generation_model_name_or_path', default=PROJECT_DIR+"/qwen2.5-3b", type=str,
                        help="model name")
    parser.add_argument("--output_dir", default=PROJECT_DIR+'/Dialogue_Planning_LLM/sft', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", default='./data', type=str,
                        help="The data directory.")
    parser.add_argument("--cache_dir", default='./storage_fast/plm', type=str,
                        help="The cache directory.")
    parser.add_argument("--do_train", default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True,
                        help="Whether to run eval.")
    parser.add_argument('--overwrite_output_dir', default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', default=True,
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_new_tokens", default=50, type=int)
    parser.add_argument('--seed', type=int, default=13,
                        help="random seed for initialization")
    parser.add_argument('--gpu', default="0 1", type=str,
                        help="Use CUDA on the device.")
    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--dp_learning_rate", default=1e-5, type=float,#5
                        help="The initial learning rate for Adam.")
    parser.add_argument("--pg_learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="DDP requirement.")
    parser.add_argument("--train_process", default=["sft_pg"], type=list,
                        help="current training process")
    parser.add_argument("--RAG", default=True, type=bool,
                        help="if add knowledge")
    parser.add_argument("--state_attention", default=True, type=bool,
                        help="if add four state aspects")
    parser.add_argument("--if_float16", default=False, type=bool,
                        help="if True, load generation_model in bfloat16 to save memory while keeping precision")
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
    is_ddp = setup_ddp(args)

    if not is_ddp:
        device, device_id = utils.set_cuda(args)
        args.device = device
        args.device_id = device_id
    else:
        args.device_id = [args.local_rank]

    print(args.device,flush=True)
    print(args.device_id, flush=True)
    # Set seed
    utils.set_random_seed(args.seed)

    config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case,
                                                     cache_dir=args.cache_dir)
    generation_config = cfg[args.generation_model_name].from_pretrained(args.generation_model_name_or_path, cache_dir=args.cache_dir, trust_remote_code=True)
    generation_tokenizer = tok[args.generation_model_name].from_pretrained(args.generation_model_name_or_path,
                                                         do_lower_case=args.do_lower_case, trust_remote_code=True)
    if "sft_dp" in args.train_process:
        dialogue_planner = MDDP(args, config, tokenizer)
        dialogue_planner.to(args.device)
    else:
        dialogue_planner=None
    if "sft_pg" in args.train_process:
        if args.if_float16:
            generation_model = AutoModelForCausalLM.from_pretrained(
                args.generation_model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        else:
            generation_model = AutoModelForCausalLM.from_pretrained(
                args.generation_model_name_or_path,
                trust_remote_code=True
            )
        generation_model.to(args.device)
        for param in generation_model.parameters():
            param.requires_grad = False
        prompt_generator = PG(args, generation_config, generation_tokenizer)
        prompt_generator.to(args.device)
    else:
        generation_model=None
        prompt_generator=None
    train_dataset = data_reader.load_and_cache_examples(args, tokenizer, generation_tokenizer,evaluate=False)

    eos_token_id = getattr(generation_tokenizer, 'eos_token_id', 2)
    if generation_tokenizer.pad_token_id is not None:
        pad_token_id=generation_tokenizer.eos_token_id
        print(pad_token_id,flush=True)
    else:
        pad_token_id = generation_tokenizer.eos_token_id
    print(f"[sft] generation model: {args.generation_model_name}, eos_token_id: {eos_token_id}", flush=True)

    logging.info("Training/evaluation parameters %s", args)
    output_dir = os.path.join(args.output_dir, 'best_checkpoint')

    # Training
    if args.do_train:
        global_step, tr_str_loss,tr_gen_loss = train(args, train_dataset, dialogue_planner,
                                                     prompt_generator, tokenizer,generation_model,
                                                     generation_tokenizer, eos_token_id=eos_token_id,pad_token_id=pad_token_id)
        logging.info(" global_step = %s, average str loss = %s,average gen loss = %s", global_step, tr_str_loss,tr_gen_loss)
        tokenizer.save_pretrained(output_dir)
    if args.do_eval:
        # Load a trained model and vocabulary that you have fine-tuned
        if "sft_dp" in args.train_process:
            dialogue_planner_output_dir = os.path.join(output_dir, 'dialogue_planner')
            if hasattr(dialogue_planner, 'module'):
                dialogue_planner.module.load_state_dict(torch.load(os.path.join(dialogue_planner_output_dir, 'pytorch_model.bin')))
            else:
                dialogue_planner.load_state_dict(torch.load(os.path.join(dialogue_planner_output_dir, 'pytorch_model.bin')))
        if "sft_pg" in args.train_process:
            prompt_generator_output_dir = os.path.join(output_dir, 'prompt_generator')
            if hasattr(prompt_generator, 'module'):
                prompt_generator.module.load_state_dict(torch.load(os.path.join(prompt_generator_output_dir, 'pytorch_model.bin')))
            else:
                prompt_generator.load_state_dict(torch.load(os.path.join(prompt_generator_output_dir, 'pytorch_model.bin')))
        generation_tokenizer = tok[args.generation_model_name].from_pretrained(args.generation_model_name_or_path, trust_remote_code=True)
        tokenizer = tok[args.model_name].from_pretrained(output_dir, do_lower_case=args.do_lower_case)
        args.set_name = 'test'
        evaluate(args, dialogue_planner,prompt_generator, tokenizer,generation_model,generation_tokenizer, save_output=True, eos_token_id=eos_token_id,pad_token_id=pad_token_id)


if __name__ == "__main__":
    main()