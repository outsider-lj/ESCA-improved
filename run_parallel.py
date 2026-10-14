from env import Env
from agent_corrected_ppo import MDDP, PG
from utils import *
from itertools import count
from tqdm import tqdm
import argparse
from transformers import BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig, AutoTokenizer, AutoConfig, \
    AutoModelForCausalLM
from fastchat.model import add_model_args
from sentence_transformers import SentenceTransformer
from data.prompt import behaviors, stages
from data_reader import clean_knowledge
from prompt import ESConvAct

# EOS/BOS 不再硬编码，统一从 tokenizer 获取
# EOS_TOKEN_ID = 2
# BOS_TOKEN_ID = 1
PROJECT_DIR = "/projects/scc/UGOE/UXEI/UMIN/scc_umin_ag_xiaoming_fu/umin_kurs_datascismartcity_s26/dir.project/lj/LLM"

tok = {'roberta': RobertaTokenizer,'llama2':AutoTokenizer, 'qwen3': AutoTokenizer,'qwen2.5': AutoTokenizer}
cfg = {'roberta': RobertaConfig, 'llama2':AutoConfig,'qwen3': AutoConfig,'qwen2.5': AutoConfig}
import string


def clean_knowledge(text):
    # 保留英文字母（a-zA-Z）、标点符号和空格
    allowed = string.ascii_letters + string.punctuation + ' '
    return ''.join(ch for ch in text if ch in allowed)


def train(args, config, dataset, filename, tokenizer, embed_model):
    env = Env(args, dataset, mode='train')  # env init
    set_random_seed(args.seed)
    dialog_planner = MDDP(args, config, tokenizer)  # policy network init
    if args.PG == True:
        if 'vicuna' in [args.system] or 'llama2' in [args.system] or 'qwen3' in [args.system] or 'qwen2.5' in [args.system]:
            generation_config = AutoConfig.from_pretrained(args.model_path)
            prompt_generator = PG(args, generation_config, env.vicuna_tokenizer)
    else:
        print("No bridge model for prompt generation")
    # load policy parameters
    if args.dp_sft_dir is not None:
        print('Staring loading policy model from {}'.format(args.dp_sft_dir))
        dialog_planner.load_model(data_name=args.data_name, filename=args.dp_sft_dir)
        # ===== dialog_planner 加载到 GPU 0 =====
        dialog_planner.to(args.dp_device)
        print(next(dialog_planner.parameters()).device, flush=True)

    if args.load_rl_epoch > 0:
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        dialog_planner.load_model(data_name=args.data_name, filename=args.dp_sft_dir, epoch_user=args.load_rl_epoch)
        # ===== dialog_planner 加载到 GPU 0 =====
        dialog_planner.to(args.dp_device)

    if args.pg_sft_dir is not None and args.PG == True:
        print('Staring loading pg model from {}'.format(args.pg_sft_dir))
        prompt_generator.load_model(data_name=args.data_name, filename=args.pg_sft_dir)
        # ===== prompt_generator 加载到 GPU 1 =====
        prompt_generator.to(args.pg_device)
        if args.if_float16 == True:
            prompt_generator.to(torch.bfloat16)
        print(next(prompt_generator.parameters()).device, flush=True)
    else:
        prompt_generator = None

    if args.load_rl_epoch > 0 and args.PG == True and prompt_generator is not None:
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        prompt_generator.load_model(data_name=args.data_name, filename=args.pg_sft_dir, epoch_user=args.load_rl_epoch)
        # ===== [修改] prompt_generator 加载到 GPU 1 =====
        prompt_generator.to(args.pg_device)

    test_performance = []
    if args.do_eval:
        dialog_planner.eval()
        if prompt_generator is not None:
            prompt_generator.eval()
        SR15_mean = evaluate(args, dataset, dialog_planner, prompt_generator, filename, 0, env, embed_model)
        test_performance = [SR15_mean]
    if not args.do_train:
        return
    else:
        dialog_planner.train()
        if prompt_generator is not None:
            prompt_generator.train()
    for train_step in range(1, args.max_steps + 1):
        SR, AvgT, total_reward = 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.dp_device)
        response_loss = torch.tensor(0, dtype=torch.float, device=args.pg_device) if args.PG else None
        # 梯度累积计数器
        dp_accum = 0
        pg_accum = 0
        # 初始 reset，准备积累数据
        dialog_planner.reset()
        if prompt_generator is not None:
            prompt_generator.reset()
        for i_episode in tqdm(range(args.sample_times), desc='sampling'):
            # blockPrint()
            print('\n================new tuple:{}===================='.format(i_episode))
            conversation, states = env.reset()
            epi_reward = 0
            for t in count():  # user  dialog
                if args.dp_rl == True:
                    action, _, _ = dialog_planner.select_action(conversation, int(states[-1]['emotion']) - 1,
                                                                int(states[-1]['trust']) - 1,
                                                                stages.index(states[-1]['stage']),
                                                                behaviors.index(states[-1]['behavior']),
                                                                train_mode='rl')  # use MDDP to predict action
                else:
                    action = dialog_planner.select_action(conversation, int(states[-1]['emotion']) - 1,
                                                          int(states[-1]['trust']) - 1,
                                                          stages.index(states[-1]['stage']),
                                                          behaviors.index(states[-1]['behavior']),
                                                          is_test=True)
                if args.RAG == True:
                    category = get_strategy_category(action)
                    if category == "information-related":
                        k = search_faiss_and_fetch_text(embed_model, conversation[-4:], args.psyqa_index_path,
                                                        args.psyqa_db_path)
                        k = clean_text(k)
                        k = summary_knowledge(category, k, conversation[-4:], env.vicuna_model, env.vicuna_tokenizer,
                                              max_new_tokens=200,
                                              temperature=0.7, model_name=args.system, device=env.vicuna_device)
                        retrieved_knowledge = "The " + category + " knowledge: " + k
                    elif category == "context-related":
                        k = generate_context_knowledge(conversation[-4:], env.vicuna_model, env.vicuna_tokenizer,
                                                       max_new_tokens=200,
                                                       temperature=0.7, model_name=args.system, device=env.vicuna_device)
                        retrieved_knowledge = "The " + category + " knowledge: " + k
                    else:
                        "Something wrong with category"
                else:
                    retrieved_knowledge = None
                if args.PG == True and args.RAG == True:
                    retrieved_knowledge = clean_knowledge(retrieved_knowledge)
                    dial = ' '.join([f"{turn['role']}: {turn['content']}" for turn in conversation])
                    if args.system=="qwen3" or args.system=="qwen2.5":
                        bos_id = env.vicuna_tokenizer.convert_tokens_to_ids("<|im_start|>")
                        eos_id = env.vicuna_tokenizer.convert_tokens_to_ids("<|im_end|>")
                    elif args.system=="llama2":
                        bos_id = env.vicuna_tokenizer.bos_token_id
                        eos_id = env.vicuna_tokenizer.eos_token_id
                    gen_input_ids = env.vicuna_tokenizer.encode(dial, add_special_tokens=False)
                    gen_input_ids = [bos_id]  + gen_input_ids[-args.max_seq_length-1:] + [eos_id]
                    gen_input_ids = torch.tensor([gen_input_ids]).long().to(args.pg_device)
                    knowledge_ids = [bos_id] + env.vicuna_tokenizer.encode(retrieved_knowledge,add_special_tokens=False)[
                        :args.max_seq_length] + [eos_id]
                    knowledge_ids = torch.tensor([knowledge_ids]).long().to(args.pg_device)
                    instruction = ESConvAct[action]
                    instruction_ids = [bos_id] + env.vicuna_tokenizer.encode(instruction,add_special_tokens=False) + [eos_id]
                    instruction_ids = torch.tensor([instruction_ids]).long().to(args.pg_device)
                    with torch.no_grad():
                        context_embeddings = env.vicuna_model.model.embed_tokens(
                            gen_input_ids)  # [1, seq_len, hidden_size]
                        knowledge_embeddings = env.vicuna_model.model.embed_tokens(knowledge_ids)
                        instruction_embeddings = env.vicuna_model.model.embed_tokens(instruction_ids)
                        First_tokens = env.vicuna_tokenizer.encode("Supporter:", add_special_tokens=False)
                        First_embedding = env.vicuna_model.model.embed_tokens(
                            torch.tensor([First_tokens]).long().to(args.pg_device))
                    target_dtype = torch.bfloat16 if args.if_float16 else torch.float32
                    context_embeddings = context_embeddings.to(args.pg_device, dtype=target_dtype)
                    knowledge_embeddings = knowledge_embeddings.to(args.pg_device, dtype=target_dtype)
                    instruction_embeddings = instruction_embeddings.to(args.pg_device, dtype=target_dtype)
                    First_embedding = First_embedding.to(args.pg_device, dtype=target_dtype)
                    gen_attention_mask = gen_input_ids.ne(eos_id).to(args.pg_device)
                    knowledge_attention_mask = knowledge_ids.eq(eos_id).to(args.pg_device)
                    instruction_attention_mask = instruction_ids.eq(eos_id).to(args.pg_device)
                    if args.pg_rl == True:
                        embedding, _, _ = prompt_generator(
                            context_embeddings=context_embeddings,
                            knowledge_embeddings=knowledge_embeddings,
                            instruction_embeddings=instruction_embeddings,
                            attention_mask=gen_attention_mask,
                            knowledge_mask=knowledge_attention_mask,
                            instruction_mask=instruction_attention_mask,
                            train_mode='rl'
                        )
                    else:
                        with torch.no_grad():
                            embedding = prompt_generator(
                                context_embeddings=context_embeddings,
                                attention_mask=None,
                                knowledge_embeddings=knowledge_embeddings,
                                instruction_embeddings=instruction_embeddings,
                                train_mode='test'
                            )
                    prompt_embedding = torch.cat([embedding, First_embedding], dim=1)
                else:
                    prompt_embedding = None
                    # pg_response = None
                conversation, response_reward = env.step(action, retrieved_knowledge,
                                                         prompt_embedding)
                if args.dynamic_state_example == True:
                    state_example_ebs, state_example_trust = search_faiss_and_fetch_text_for_state(embed_model,
                                                                                                   conversation[-4:],
                                                                                                   args.esc_index_path,
                                                                                                   args.esc_db_path,
                                                                                                   top_k=1)
                    state_example_ebs = " ".join(state_example_ebs)
                    state_example_trust = " ".join(state_example_trust[0])
                else:
                    state_example_ebs = (
                        "Input: seeker: I've been reading about ways to manage anxiety. I found some breathing techniques online and tried one yesterday. supporter: That's a great first step. How did it go? seeker: It actually helped a little. I think I want to keep trying different things to see what works for me. "
                        "Output: negative emotional intensity: 2, behavior: answer, stage of change:preparation ")
                    state_example_trust = (
                        "Input: seeker: Lately, I've been feeling so exhausted. No matter how much I sleep, I still wake up tired. supporter: That sounds really tough. Have you been feeling this way for a long time? seeker: Yeah, it's been weeks now. I don't know if it's just stress or something worse, but I feel like I have no energy for anything. "
                        "Output: reliability: 3, response competence: 4,perceived understandability: 4")
                states, reward, critic_score,done = env.state_step(state_example_ebs, state_example_trust)
                # state tracking and self-play
                epi_reward += reward + response_reward
                if args.dp_rl == True:
                    # ===== [修改] planner reward 放到 dp_device (GPU 0) =====
                    planner_reward = torch.tensor([reward], device=args.dp_device, dtype=torch.float)
                    dialog_planner.rewards.append(planner_reward)
                if args.pg_rl == True:
                    response_reward = response_reward# +0.3*reward

                    # ===== [修改] response reward 放到 pg_device (GPU 1) =====
                    response_reward = torch.tensor([response_reward], device=args.pg_device, dtype=torch.float)
                    prompt_generator.rewards.append(response_reward)
                if done:
                    if done == 1:
                        SR += 1
                    AvgT += t + 1
                    total_reward += epi_reward
                    break
            # 非交替：各自独立累积，达阈值即优化
            if args.dp_rl:
                dp_accum += 1
                if dp_accum >= args.accum_steps:
                    dialog_planner.optimize_model(train_mode='rl')
                    dp_accum = 0
                    dialog_planner.reset()
            if args.pg_rl:
                pg_accum += 1
                if pg_accum >= args.accum_steps:
                    prompt_generator.optimize_PPO()
                    pg_accum = 0
                    prompt_generator.reset()

        enablePrint()  # Enable print function
        print('loss : {} in epoch_uesr {}'.format(loss.item() / args.sample_times, args.sample_times))
        if response_loss is not None:
            print('response_loss : {} in epoch_uesr {}'.format(response_loss.item() / args.sample_times, args.sample_times))
        print('SR:{}, AvgT:{}, rewards:{} Total epoch_uesr:{}'.format(SR / args.sample_times,
                                                                      AvgT / args.sample_times,
                                                                      total_reward / args.sample_times,
                                                                      args.sample_times))
        if train_step % args.save_num == 0:
            if args.dp_rl == True:
                dialog_planner.save_model(data_name=args.data_name, filename=filename + "dp", epoch_user=train_step)
            if args.pg_rl == True and prompt_generator is not None:
                prompt_generator.save_model(data_name=args.data_name, filename=filename + "pg", epoch_user=train_step)
        if train_step % args.eval_num == 0:
            dialog_planner.eval()
            if prompt_generator is not None:
                prompt_generator.eval()
            SR_all = evaluate(args, dataset, dialog_planner, prompt_generator, filename, train_step, env, embed_model)
            test_performance.append(SR_all)
            dialog_planner.train()
            if prompt_generator is not None:
                prompt_generator.train()
    print(test_performance)


def evaluate(args, dataset, dialogue_planner, prompt_generator, filename, i_episode, train_env, embed_model):
    if any(m in [args.system, args.user, args.critic] for m in ['vicuna', 'llama2', 'qwen3','qwen2.5']):
        test_env = Env(args, dataset, mode='test', env_model=train_env.vicuna_model,
                       env_tokenizer=train_env.vicuna_tokenizer)
    else:
        test_env = Env(args, dataset, mode='test')  # env init
    test_env.vicuna_model = test_env.vicuna_model
    set_random_seed(args.seed)

    SR, AvgT, total_reward = 0, 0, 0
    SR_turn = [0] * args.max_turn
    test_size = len(test_env.dataset)
    print('Test size: ', test_size)
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    record_filename = 'Record-epoch-{}-'.format(i_episode) + filename
    REC_PATH = TMP_DIR[args.data_name] + '/eval_result/' + record_filename + '.txt'
    if not os.path.isdir(TMP_DIR[args.data_name] + '/eval_result/'):
        os.makedirs(TMP_DIR[args.data_name] + '/eval_result/')
    rec_file = open(REC_PATH, 'w')
    for test_num in tqdm(range(test_size)):  # test_size
        # blockPrint()
        print('\n================test tuple:{}===================='.format(test_num))
        epi_reward = 0
        done = 0
        is_last_turn = False
        conversation, states = test_env.reset()
        if args.system == "qwen3" or args.system == "qwen2.5":
            bos_id = test_env.vicuna_tokenizer.convert_tokens_to_ids("<|im_start|>")
            eos_id = test_env.vicuna_tokenizer.convert_tokens_to_ids("<|im_end|>")
        elif args.system == "llama2":
            bos_id = test_env.vicuna_tokenizer.bos_token_id
            eos_id = test_env.vicuna_tokenizer.eos_token_id
        for t in count():  # user  dialog
            if args.state_attention == True:
                emotion = int(states[-1]['emotion']) - 1
                trust = int(states[-1]['trust']) - 1
                stage = stages.index(states[-1]['stage'])
                behavior = behaviors.index(states[-1]['behavior'])
            else:
                emotion = None
                trust = None
                stage = None
                behavior = None
            action = dialogue_planner.select_action(conversation, emotion, trust, stage, behavior,
                                                    is_test=True)  # use MDDP to predict action
            # ===== [修改] 提前定义 target_dtype，确保 embedding 与 prompt_generator dtype 一致 =====
            target_dtype = torch.bfloat16 if args.if_float16 else torch.float32
            if args.RAG == True:
                category = get_strategy_category(action)
                if category == "information-related":
                    k = search_faiss_and_fetch_text(embed_model, conversation[-4:], args.psyqa_index_path,
                                                    args.psyqa_db_path)
                    k = clean_text(k)
                    k = summary_knowledge(category, k, conversation[-4:], test_env.vicuna_model,
                                          test_env.vicuna_tokenizer,
                                          max_new_tokens=200,
                                          temperature=0.7, model_name=args.system, device=test_env.vicuna_device)
                    retrieved_knowledge = "The " + category + " knowledge: " + k
                    retrieved_knowledge = clean_knowledge(retrieved_knowledge)
                elif category == "context-related":
                    k = generate_context_knowledge(conversation[-4:], test_env.vicuna_model, test_env.vicuna_tokenizer,
                                                   max_new_tokens=200,
                                                   temperature=0.7, model_name=args.system, device=test_env.vicuna_device)
                    retrieved_knowledge = "The " + category + " knowledge: " + k
                    retrieved_knowledge = clean_knowledge(retrieved_knowledge)
                else:
                    "Something wrong with category"
                retrieved_knowledge = clean_knowledge(retrieved_knowledge)
                if args.PG == True:
                    # ===== [修改] knowledge_ids 发送到 pg_device (GPU 1) =====
                    knowledge_ids = ([bos_id] if bos_id is not None else []) + test_env.vicuna_tokenizer.encode(retrieved_knowledge,add_special_tokens=False)[
                        :args.max_seq_length] + ([eos_id] if eos_id is not None else [])
                    knowledge_ids = torch.tensor([knowledge_ids]).long().to(args.pg_device)
                    with torch.no_grad():
                        knowledge_embeddings = test_env.vicuna_model.model.embed_tokens(knowledge_ids)
                    # ===== [修改] 确保 knowledge_embeddings 在 pg_device 上，并与 prompt_generator dtype 一致 =====
                    knowledge_embeddings = knowledge_embeddings.to(args.pg_device, dtype=target_dtype)
                else:
                    knowledge_embeddings = None
            else:
                retrieved_knowledge = " "
            print(retrieved_knowledge, flush=True)
            if args.PG == True:
                dial = ' '.join([f"{turn['role']}: {turn['content']}" for turn in conversation])
                gen_input_ids = test_env.vicuna_tokenizer.encode(dial,add_special_tokens=False)
                gen_input_ids =([bos_id] if bos_id is not None else [])+ gen_input_ids[-args.max_seq_length + 1:] + ([eos_id] if eos_id is not None else [])
                gen_input_ids = torch.tensor([gen_input_ids]).long().to(args.pg_device)
                instruction = ESConvAct[action]
                instruction_ids = ([bos_id] if bos_id is not None else []) + test_env.vicuna_tokenizer.encode(instruction,add_special_tokens=False) + ([eos_id] if eos_id is not None else [])
                instruction_ids = torch.tensor([instruction_ids]).long().to(args.pg_device)
                with torch.no_grad():
                    context_embeddings = test_env.vicuna_model.model.embed_tokens(
                        gen_input_ids)  # [1, seq_len, hidden_size]
                    instruction_embeddings = test_env.vicuna_model.model.embed_tokens(instruction_ids)
                    First_tokens = test_env.vicuna_tokenizer.encode("Supporter:", add_special_tokens=False)
                    First_embedding = test_env.vicuna_model.model.embed_tokens(
                        torch.tensor([First_tokens]).long().to(args.pg_device))
                context_embeddings = context_embeddings.to(args.pg_device, dtype=target_dtype)
                instruction_embeddings = instruction_embeddings.to(args.pg_device, dtype=target_dtype)
                First_embedding = First_embedding.to(args.pg_device, dtype=target_dtype)
                gen_attention_mask = gen_input_ids.ne(eos_id).to(args.pg_device)
                if args.RAG==False:
                    knowledge_attention_mask=None
                    knowledge_embeddings=None
                else:
                    knowledge_attention_mask = knowledge_ids.eq(eos_id).to(args.pg_device)
                instruction_attention_mask = instruction_ids.eq(eos_id).to(args.pg_device)
                embedding = prompt_generator(
                    context_embeddings=context_embeddings,
                    knowledge_embeddings=knowledge_embeddings,
                    instruction_embeddings=instruction_embeddings,
                    attention_mask=gen_attention_mask,
                    knowledge_mask=knowledge_attention_mask,
                    instruction_mask=instruction_attention_mask,
                    train_mode="test"
                )
                prompt_embedding = torch.cat([embedding, First_embedding], dim=1)
            else:
                prompt_embedding = None
            conversation, response_reward = test_env.step(action, retrieved_knowledge,
                                                          prompt_embedding)
            if args.dynamic_state_example == True:
                state_example_ebs, state_example_trust = search_faiss_and_fetch_text_for_state(embed_model,
                                                                                               conversation[-4:],
                                                                                               args.esc_index_path,
                                                                                               args.esc_db_path,
                                                                                               top_k=1)
                state_example_ebs = " ".join(state_example_ebs)
                state_example_trust = " ".join(state_example_trust[0])
            else:
                state_example_ebs = (
                    "Input: Seeker: I've been reading about ways to manage anxiety. I found some breathing techniques online and tried one yesterday. Supporter: That's a great first step. How did it go? seeker: It actually helped a little. I think I want to keep trying different things to see what works for me. "
                    "Output: negative emotional intensity: 2, behavior: answer, stage of change:preparation ")
                state_example_trust = (
                    "Input: Seeker: Lately, I've been feeling so exhausted. No matter how much I sleep, I still wake up tired. Supporter: That sounds really tough. Have you been feeling this way for a long time? seeker: Yeah, it's been weeks now. I don't know if it's just stress or something worse, but I feel like I have no energy for anything. "
                    "Output: reliability:3, response competence:4,perceived understandability:4")
            states, reward, critic_score,done = test_env.state_step(state_example_ebs, state_example_trust)#reward为state reward
            if args.data_name == 'cb' and reward < 0:  # reward = Sale-to-List Ratio
                reward = 0
            epi_reward += reward + response_reward
            if done:
                if done == 1:
                    SR_turn = [v + 1 if i > t else v for i, v in enumerate(SR_turn)]
                    SR += 1
                total_reward += epi_reward
                AvgT += t + 1
                rec_file.write('%s\n\n' % str({'dialog': states[-1], 'reward': epi_reward}))
                break
        enablePrint()

    SR_mean = float(SR) / test_size
    AvgT_mean = float(AvgT) / test_size
    reward_mean = total_reward / test_size
    SR_all = [SR_mean, AvgT_mean, reward_mean]
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=test_num, SR=SR_all, mode='test')  # save RL SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = float(SR_turn[i]) / test_size
    print('success turn:{}'.format(SRturn_all))
    print('SR:{}, AvgT:{}, reward:{}'.format(SR_mean, AvgT_mean, reward_mean))
    PATH = TMP_DIR[args.data_name] + '/eval_result/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(test_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')
    with open(PATH, 'a') as f:
        f.write('{}\t{}\t{}\t{}\n'.format(i_episode, SR_mean, AvgT_mean, reward_mean))
    return SR_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=13, help='random seed.')
    parser.add_argument('--num_gpus', type=int, default=2, help='number of gpus.')
    # parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--dp_learning_rate', type=float, default=1e-6, help='learning rate.')
    parser.add_argument('--pg_learning_rate', type=float, default=1e-6, help='learning rate.')
    parser.add_argument('--data_name', type=str, default='esc', choices=['esc', 'cima', 'cb'],
                        help='One of {esc}.')
    parser.add_argument('--system', type=str, default='llama2', choices=['vicuna', 'chatgpt', 'llama2', 'chat_ai', 'qwen3', 'qwen2.5'],
                        help='One of {vicuna, chatgpt, llama2, chat_ai, qwen3}.')
    parser.add_argument('--user', type=str, default='chat_ai', choices=['vicuna', 'chatgpt', 'llama2', 'chat_ai', 'qwen3'],
                        help='One of {vicuna, chatgpt, llama2, chat_ai, qwen3}.')
    parser.add_argument('--critic', type=str, default='chat_ai', choices=['vicuna', 'chatgpt', 'llama2', 'chat_ai', 'qwen3'],
                        help='One of {vicuna, chatgpt, llama2, chat_ai, qwen3}.')
    parser.add_argument('--dp_sft_dir',
                        default=PROJECT_DIR + '/Dialogue_Planning_LLM/sft/esc/roberta/Truestate_attention/Truerag/best_checkpoint/dialogue_planner',
                        # ../pretrain/outputs/best_pretrain.pt#'/mnt/vast-standard/home/jli6/u27481/Dialogue_Planning_LLM/tmp/esc/RL-agent/esc-llama2-chat_ai-chat_aidp-epoch-2',#
                        type=str,
                        help="Pretrain model path.")  # sft/esc/roberta/Truestate_attention/Truerag/best_checkpoint/dialogue_planner#
    parser.add_argument('--pg_sft_dir',
                        default=PROJECT_DIR + '/Dialogue_Planning_LLM/sft/esc/roberta/Truestate_attention/Truerag/best_checkpoint/prompt_generator',
                        #'/mnt/vast-standard/home/jli6/u27481/Dialogue_Planning_LLM/tmp/esc/RL-agent/esc-llama2-chat_ai-chat_aipg-epoch-2',#
                        type=str, help="Pretrain model path.")

    parser.add_argument("--psyqa_index_path", type=str, default="./knowledge/psyqa_knowledge_faiss.index",
                        help="Path to FAISS index for PsyQA")
    parser.add_argument("--psyqa_db_path", type=str, default="./knowledge/psyqa_knowledge.db",
                        help="CSV file for PsyQA knowledge")
    parser.add_argument("--esc_index_path", type=str, default="./knowledge/esconv_knowledge_faiss.index",
                        help="Path to FAISS index for Emotion-Reflection")
    parser.add_argument("--esc_db_path", type=str, default="./knowledge/esconv_knowledge.db",
                        help="CSV file for Emotion-Reflection")
    parser.add_argument('--max_turn', type=int, default=10, help='max conversation turn')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='load agent from epoch')

    parser.add_argument("--cache_dir", default='./storage_fast/plm', type=str, help="The cache directory.")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--model_path", type=str, default=PROJECT_DIR + "/llama2-7b-chat-hf")#/llama2-7b-chat-hf
    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument("--model_name_or_path", default=PROJECT_DIR + '/roberta-large', type=str,
                        help="model name or path")  # /Users/lijing/Documents/博士/code
    parser.add_argument("--chat_ai_model", type=str,
                        default="deepseek-v4-pro")  # qwen2.5-72b-instruct#"llama-3.3-70b-instruct" #meta-llama-3.1-70b-instruct #deepseek-r1-distill-llama-70b
    parser.add_argument("--do_lower_case", default=True, help="Set this flag if you are using an uncased model.")
    parser.add_argument('--max_steps', type=int, default=5, help='max training steps')
    parser.add_argument('--sample_times', type=int, default=80, help='the epoch of sampling')
    parser.add_argument('--accum_steps', type=int, default=5, help='gradient accumulation: episodes per PPO update')
    parser.add_argument('--eval_num', type=int, default=1, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=1, help='the number of steps to save RL model and metric')
    # RAG
    parser.add_argument("--embed_model_path", type=str, default=PROJECT_DIR + '/all-MiniLM-L6-v2')
    parser.add_argument("--RAG", type=bool, default=True)
    parser.add_argument("--PG", type=bool, default=False)
    parser.add_argument("--state_attention", default=True, type=bool,
                        help="if add four state aspects")
    # 强化学习需要优化两个模型,一个是MDDP，一个是PG
    parser.add_argument("--do_train", default=False, help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, help="Whether to run eval.")
    parser.add_argument('--eps_clip', type=float, default=0.2, help='clip')
    parser.add_argument("--dp_rl", type=bool, default=True, help="Whether to run RL training for MDDP model.")
    parser.add_argument("--pg_rl", type=bool, default=True, help="Whether to run RL training for PG model.")
    parser.add_argument("--dynamic_state_example", type=bool, default=True,
                        help="Whether to run RL training for PG model.")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--if_float16', type=bool, default=False)

    add_model_args(parser)
    args = parser.parse_args()
    print(args)

    # ===== 双 GPU 设备分配 =====
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

    dataset = load_dataset(args.data_name)
    filename = '{}-{}-{}-{}'.format(args.data_name, args.system, args.user, args.critic)
    print(filename, flush=True)
    config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case,
                                                     cache_dir=args.cache_dir)

    # if args.sft_dir:
    #     args.sft_dir = os.path.join(args.sft_dir, args.data_name, args.model_name, 'best_checkpoint')
    # if not os.path.exists(args.dp_sft_dir):
    #     print("no dp sft model, randomly initialize dp model")
    #     args.df_sft_dir = None
    # if not os.path.exists(args.pg_sft_dir):
    #     print("no pg sft model, randomly initialize PG model")
    #     args.pg_sft_dir = None
    embed_model = SentenceTransformer(args.embed_model_path)
    train(args, config, dataset, filename, tokenizer, embed_model)


if __name__ == '__main__':
    main()
