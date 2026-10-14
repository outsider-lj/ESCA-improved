from transformers import AutoTokenizer, AutoModelForCausalLM
from data.prompt import behaviors,stages
from fastchat.model import load_model, get_conversation_template

from utils import *
from prompt import *
import nltk
import re
import time
from data.chek_data import average_to_5_level
system_role = {'esc':'Supporter', 'cima': 'Teacher', 'cb': 'Buyer'}
user_role = {'esc':'Seeker', 'cima': 'Student', 'cb': 'Seller'}
message_format = {'esc': ESConvMessages}

YOUR_API_KEY = ""

import faiss
import sqlite3
def search_faiss_and_fetch_text(knowledge_type,encoder, dial_history, index_path, db_path, top_k=2):
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
        cursor.execute("SELECT context,emotion,trust,behavior,stage FROM knowledge WHERE faiss_index=?", (int(idx),))
        row = cursor.fetchone()
        if row:
            context=row[0]
            emotion=row[1]
            trust=row[2]
            behavior=row[3]
            stage=row[4]
            result=str(context+" emotional intensity: "+emotion+" trust: "+trust+" behavior: "+behavior+"stage of change: "+stage)
            results.append(result)
    conn.close()
    return results

class Env(object):
    def __init__(self, args, dataset, mode, env_model=None, env_tokenizer=None):
        self.vicuna_device = getattr(args, 'pg_device', torch.device('cuda:0'))
        if any(m in [args.system, args.user, args.critic] for m in ['vicuna', 'llama2', 'qwen3','qwen2.5']):
            if mode == 'train':
                try:
                    self.vicuna_model = AutoModelForCausalLM.from_pretrained(
                        args.model_path,
                        trust_remote_code=True,
                    ).to(self.vicuna_device).eval()
                except (TypeError, ValueError):
                    self.vicuna_model = AutoModelForCausalLM.from_pretrained(
                        args.model_path,
                    ).to(self.vicuna_device).eval()
                if args.if_float16==True:
                    self.vicuna_model.to(torch.bfloat16)
                try:
                    self.vicuna_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
                except (TypeError, ValueError):
                    self.vicuna_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                for param in self.vicuna_model.parameters():
                    param.requires_grad = False
            else:
                self.vicuna_model = env_model.to(self.vicuna_device).eval()
                self.vicuna_tokenizer = env_tokenizer
        if not hasattr(args, 'eos_token_id') or args.eos_token_id is None:
            args.eos_token_id = self.vicuna_tokenizer.eos_token_id
        self.args = args
        self.dataset = dataset[mode]
        self.max_turn = args.max_turn
        self.conversation = []
        self.cur_conver_step = 0
        self.test_num = 0
        self.mode = mode
        self.states=[]
        self.reward_dict = {
            'esc': {
                'worse': -1.0,
                'same': -0.5,
                'better': 0.5,
                'solved': 1.0,
            },
        }
        self.response_reward_dict = {
                'not': -1.0,
                'poor': 0.5,
                'good': 1.0,
        }
        self.past_emotions = []
        self.past_trusts = []
        self.N = 2  # 计算奖励的滑动窗口大小
        # set_random_seed(args.seed)

    def strip_user_turn(self, response):
        """去掉回复中混入的「用户角色:」及之后的内容。

        例如模型生成：
          "I know it's hard to feel this way, but you need to remember
           that you're not alone in this. Seeker: I just feel like..."
        只保留 "Seeker:" 之前的部分。
        """
        if not response:
            return response
        user_name = user_role.get(self.args.data_name, 'Seeker')
        pattern = re.compile(r'\b' + re.escape(user_name) + r'\s*[:：]')
        m = pattern.search(response)
        if m:
            response = response[:m.start()].strip()
        return response

    def reset(self):
        self.past_emotions.clear()
        self.past_trusts.clear()
        state=dict()
        self.cur_conver_step = 0
        if self.mode == 'train':
            self.case = np.random.choice(self.dataset)
        elif self.mode == 'test':
            self.case = self.dataset[self.test_num]
            self.test_num += 1
        
        if self.args.data_name == 'esc':
            self.conversation = [{"role":"Seeker", "content":self.case['situation']}]
        print(self.conversation,flush=True)
        state.update({"emotion":5,"behavior":"statement-emotion","stage":"precontemplation","trust":3})
        self.states=[state]
        _ = self.compute_reward(int(state["emotion"]), state["trust"])
        print(state,flush=True)
        return self.conversation,self.states


    def step(self, action,retrieved_knowledge,prompt_embedding=None):
        print('---------------step:{}-------------'.format(self.cur_conver_step))
        print(action,flush=True)
        ## 引入知识
        if prompt_embedding is None:
            messages = message_format[self.args.data_name](self.case, 'system', self.conversation, action,retrieved_knowledge)
            response = self.generate_response(self.args.system, messages, system_role[self.args.data_name])
            response = self.strip_user_turn(response)
            response = self.postprocess_response(response, system_role[self.args.data_name])
        else:
            response, generated_tokens = self.generate_response_from_query(prompt_embedding)
            response = self.strip_user_turn(response)
            response = self.postprocess_response(response, system_role[self.args.data_name])

        self.conversation.append({"role":system_role[self.args.data_name],"content":response})
        print(self.conversation[-1],flush=True)
        # 获取回复的奖励
        if self.mode != 'test' and self.args.pg_rl==True:
            messages = message_format[self.args.data_name](self.case, 'response_reward', self.conversation,action=action)
            response_reward = self.get_response_reward(
                messages,
                'response_reward',
                self.args.critic,
                response_text=response,  # 传递生成的回复文本# 传递生成的token IDs用于准确EOS检查
            )
            print(response_reward,flush=True)
        else:
            print("No response reward in test mode",flush=True)
            response_reward = 0
        messages = message_format[self.args.data_name](self.case, 'user', self.conversation)
        user_response = self.generate_response(self.args.user, messages, user_role[self.args.data_name])
        user_response = self.postprocess_response(user_response,user_role[self.args.data_name])
        self.conversation.append({"role": user_role[self.args.data_name], "content": user_response})
        print(self.conversation[-1],flush=True)

        return self.conversation, response_reward
    def state_step(self,state_example_abs,state_example_trust):
        state = dict()
        done=0
        if self.args.state_attention == True:
            state_message = ESConvMessages(self.case, "state_tracking_ebs", self.conversation,retrieved_knowledge=state_example_abs)
            state = self.get_user_states(state_message, self.args.chat_ai_model, state, "state_tracking_ebs",
                                         self.args.system)
            state_message = ESConvMessages(self.case, "state_tracking_trust", self.conversation,retrieved_knowledge=state_example_trust)
            state = self.get_user_states(state_message, self.args.chat_ai_model, state, "state_tracking_trust",
                                         self.args.system)
            self.states.append(state)
            print(self.states[-1], flush=True)
        else:
            print("Strategy attention is not enabled, skipping state tracking.", flush=True)

        # 判断是否达成目标（先算 critic_score，用于 reward 和 done）
        messages = message_format[self.args.data_name](self.case, 'critic', self.conversation)
        critic_score = self.get_critic(self.args.chat_ai_model, messages, self.args.critic)

        # 获取 reward（emotion 0.3 + trust 0.3 + critic_score 0.4）
        if self.mode != 'test' and self.args.dp_rl==True:
            reward = self.compute_reward(int(state["emotion"]), state["trust"], critic_score)
            print(reward, flush=True)
        else:
            reward = 0
            print("Reward is set to 0 in test mode", flush=True)

        if self.args.data_name == 'esc':
            # if state['emotion']==1 or state['emotion']==2:
            #     done=1
            if critic_score > 0.5:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')

        self.cur_conver_step += 1
        return self.states,reward,critic_score,done
    def postprocess_response(self, response, role):
        #print(response)
        if role in response:
            response = response.split(role)[0].strip()
        sents = nltk.sent_tokenize(response)
        if len(sents) == 1:
            if response[-1] not in ['.','!','?',':']:
                return response + '.'
            return response.strip()
        try:
            if sents[-1].strip()[-1] not in ['.','!','?',':']:
                return ' '.join(sents[:-1]).strip()
            else:
                return response.strip()
        except Exception as e:
            return response.strip()

    def generate_response_from_query(self, inputs_embeds):
        """
        生成回复，返回回复文本和生成的token IDs

        Returns:
            response: 解码后的回复文本
            generated_tokens: 生成的token IDs列表（包含EOS token如果生成了）
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        generated = []

        # 确保 inputs_embeds 与 vicuna_model 的 dtype 一致（prompt_generator 可能是 bfloat16，但 vicuna_model 是 float16）
        if inputs_embeds.dtype != next(self.vicuna_model.parameters()).dtype:
            inputs_embeds = inputs_embeds.to(dtype=next(self.vicuna_model.parameters()).dtype)

        # 1. 先跑一遍模型，得到 initial logits 和 past_key_values
        with torch.no_grad():
            outputs = self.vicuna_model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

        # 取最后一个位置的 logits，转为 float32 避免 softmax 在 float16 下溢出
            next_token_logits = (logits[:, -1, :] / self.args.temperature).float()
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
        next_token_flat = next_token.view(1)
        generated.append(next_token_flat.item())

        for _ in range(self.args.max_new_tokens - 1):
            with torch.no_grad():
                outputs = self.vicuna_model(
                    input_ids=next_token,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            next_token_logits = (logits[:, -1, :] / self.args.temperature).float()

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_flat = next_token.view(1)
            generated.append(next_token_flat.item())

            if self.args.eos_token_id is not None and (next_token_flat == self.args.eos_token_id).all():
                break

        # 解码生成文本（跳过特殊token，包括EOS）
        response = self.vicuna_tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if response.startswith('Supporter:'):
            response = response[len('Supporter:'):].lstrip()
        elif response.startswith('assistant:'):
            response = response[len('assistant:'):].lstrip()

        # 返回回复文本和生成的token IDs
        return response, generated

    def generate_response(self, model, messages, role):
        if self.mode == 'test':
            temperature = 0.7
        else:
            temperature = 0.7
        if model == 'llama2':
            prompt = llama2_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).to(self.vicuna_device),
                max_new_tokens=max_new_tokens,
                do_sample=True,          # [修复] 显式开启采样
                temperature = temperature,
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'qwen3'or model == 'qwen2.5':
            # 修复：Qwen3 的 chat template 只认 system/user/assistant，必须把 Supporter→assistant、Seeker→user 映射
            mapped_messages =chatgpt_prompt(messages, role)
            prompt = self.vicuna_tokenizer.apply_chat_template(
                mapped_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).to(self.vicuna_device),
                max_new_tokens=max_new_tokens,
                do_sample=True,          # [修复] 必须显式开启采样，否则 temperature 被忽略，走贪心解码导致回复高度一致
                temperature=temperature,
                # top_p=0.,
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'chat_ai':
            messages = chatgpt_prompt(messages, role)
            output = query_chat_ai_model(
                messages=messages,
                model=self.args.chat_ai_model,
                max_tokens=self.args.max_new_tokens,
                temperature=temperature
            )
        return output
    def get_response_reward(self, messages, role="response_reward", model="chat_ai", response_text=None):
        """
        计算回复的奖励，同时检查是否生成EOS
        
        Args:
            model_name: 模型名称
            messages: 消息列表
            role: 角色
            model: 模型类型
            response_text: 生成的回复文本，用于检查EOS
            
        Returns:
            reward: 总奖励值
        """
        if model == 'chat_ai':
            messages = chatgpt_prompt(messages, role)
            outputs = query_chat_ai_model(
                messages=messages,
                model=self.args.chat_ai_model,  # qwen2.5-72b-instruct #meta-llama-3.1-70b-instruct
                max_tokens=24,
                temperature=1.1,
                n=5
            )
            #     convert text to dict
            rewards = []
            for output in outputs:
                for key in self.response_reward_dict:
                    if key in output.lower():
                        rewards.append(self.response_reward_dict[key])
                        break
            if len(rewards) == 0:
                reward = 0
            else:
                reward = sum(rewards) / len(rewards)
        
        # 添加EOS检查逻辑 - 使用generated_tokens进行准确检查
        # eos_bonus = self.check_eos_completion(response_text, generated_tokens)
        total_reward = reward# + 0.5*eos_bonus
        
        #
        return total_reward



    def _fix_letter_spacing(self, text):
        """
        修复模型生成的逐字母+空格输出。
        如果文本中连续出现3个以上"单字母+空格"模式（如 "r e l i a b i l i t y"），
        则移除字母之间的空格，还原为连续文本。
        """
        import re
        # 检测是否存在逐字母模式：连续3个以上"字母 字母 字母"模式
        if re.search(r'\b[a-zA-Z](?: [a-zA-Z]){2,}\b', text):
            # 移除字母之间的空格
            text = re.sub(r'(?<=\b[a-zA-Z]) (?=[a-zA-Z]\b)', '', text)
            # 压缩多余空格
            text = re.sub(r' {2,}', ' ', text)
        return text

    def get_user_states(self, messages, model_name,state, role, model="llama2"):
        if model == 'chat_ai':
            print(model_name,flush=True)
            messages = chatgpt_prompt(messages, role)
            output = query_chat_ai_model(
                messages=messages,
                model=model_name,
                temperature=0.0

            )
            #     convert text to dict
            result = output.strip("**")
            result = result.strip()
            result = re.split(r':', result)
        elif model == "llama2":
            # prompt = llama2_prompt(messages, role)
            prompt = self.vicuna_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            input_ids=torch.tensor(input_ids).long().to(self.vicuna_device)
            with torch.no_grad():
                output_ids = self.vicuna_model.generate(
                    input_ids,
                    # max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True,
                    # num_return_sequences=10,
                )
            # outputs = []
            # for o in output_ids:
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True)
            # output = self._fix_letter_spacing(output)
                # outputs.append(output)
            # print(output)
            # print(output,flush=True)
            # output = output.split("\n")
            # result=output[-1]
        elif model == "qwen3" or model == 'qwen2.5':
            prompt = self.vicuna_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            input_ids = torch.tensor(input_ids).long().to(self.vicuna_device)
            with torch.no_grad():
                output_ids = self.vicuna_model.generate(
                    input_ids,
                    # max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True,
                )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
            spaces_between_special_tokens=False)
        elif model == "vicuna":
            # prompt = vicuna_prompt(messages, role)
            prompt = self.vicuna_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            input_ids = torch.tensor(input_ids).long().to(self.vicuna_device)
            # print(prompt)
            with torch.no_grad():
                output_ids = self.vicuna_model.generate(
                    torch.as_tensor(input_ids).to(self.vicuna_device),
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True,
                    # num_return_sequences=10,
                )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True)
                # output = self._fix_letter_spacing(output)
                outputs.append(output)

        if role=="state_tracking_emotion":
            # 支持 ":" 或 "="，可选括号后缀
            emotion_pattern = r'negative emotional intensity\s*[:=]\s*(\d+)'
            try:
                # try blockf 只找最后一个 block
                # output_parts = output.split("\n")
                result = output.lower()
                match = re.search(emotion_pattern, result, re.IGNORECASE)
                if match:
                    emotion = int(match.group(1))
                else:
                    raise ValueError("Primary emotion match failed")
            except:
                # fallback: 全文查找
                fallback_matches = re.findall(emotion_pattern, output.lower(), re.IGNORECASE)
                print("fallback", fallback_matches)
                if fallback_matches:
                    emotion = int(fallback_matches[-1])
                    print(f"[Fallback Match] emotion: {emotion}")
                else:
                    print(f"Filled fixed value.")
                    emotion = 3  # fallback 默认值，可自定义
            state.update({
                "emotion": emotion,
                })
        elif role=="state_tracking_behavior":
            behavior_pattern = (
                    r'behavior\s*[:=]\s*(?:\d+\s*[,()]?\s*)?'  # optional number
                    r'(?P<behavior>(' + '|'.join(behaviors) + r'))'
            )

            try:
                # output_parts = output.split("\n\n")
                result = output.lower()
                match = re.search(behavior_pattern, result, re.IGNORECASE)
                if match:
                    behavior = match.group("behavior").strip().lower()
                else:
                    raise ValueError("Primary behavior match failed")
            except:
                fallback_matches = re.findall(behavior_pattern, output, re.IGNORECASE)
                if fallback_matches:
                    print(fallback_matches,flush=True)
                    behavior = fallback_matches[-1]
                    if isinstance(behavior, tuple):
                        behavior=behavior[-1].strip().lower()
                    else:
                        behavior = behavior.strip().lower()
                    print(f"[Fallback Match] behavior: {behavior}")
                else:
                    behavior = "others"  # fallback default
                    print("[Warning] No behavior found, using default.")

            state.update({"behavior": behavior})


        elif role == "state_tracking_stage":
            stage_pattern = (
                    r'stage of change\s*[:=]\s*'
                    r'(?:\d+\s*[,()]?\s*)?'
                    r'(?P<stage>(' + '|'.join(stages) + r'))'
            )
            try:
                # output_parts = output.split("\n\n")
                result = output.lower()
                match = re.search(stage_pattern, result, re.IGNORECASE)
                if match:
                    stage = match.group("stage").strip().lower()
                else:
                    raise ValueError("Primary stage match failed")

            except:
                fallback_matches = re.findall(stage_pattern, output, re.IGNORECASE)
                if fallback_matches:
                    stage = fallback_matches[-1]
                    if isinstance(stage, tuple):
                        stage=stage[-1].strip().lower()
                    else:
                        stage = stage.strip().lower()
                    print(f"[Fallback Match] stage: {stage}")
                else:
                    stage = "contemplation"  # fallback default，可自定义
                    print("[Warning] No stage found, using default.")

            state.update({"stage": stage})
        elif role == "state_tracking_ebs":
            output=output.lower()
            print("---state-ebs---\n",flush=True)
            print(output,flush=True)
            try:


                # 方案2：简化正则表达式（推荐）
                # 不需要匹配数字前缀，直接匹配标签后的内容
                emotion_pattern = r'negative emotional intensity\s*[:=]\s*(\d)'
                # behavior_pattern = r'behavior\s*[:=]\s*(\w+)'  # 简化版
                # stage_pattern = r'stage of change\s*[:=]\s*(\w+)'

                # 或者更精确地匹配你的行为列表
                behavior_pattern = r'behavior\s*[:=]\s*(?P<behavior>\b(?:' + '|'.join(behaviors) + r'))'
                stage_pattern = r'stage of change\s*[:=]\s*(?P<stage>\b(?:' + '|'.join(stages) + r'))'

                # 完整的修改代码
                output_parts = output.split("\n")
                output_str = " ".join(output_parts)
                # 找到包含注释的行
                annotation_lines = []
                for line in output_parts:
                    if any(keyword in line.lower() for keyword in
                           ['negative emotional intensity', 'behavior', 'stage of change']):
                        annotation_lines.append(line)
                if annotation_lines:
                    result = "\n".join(annotation_lines)
                else:
                    result = output_parts[-1] if output_parts else output

                # 正则匹配
                emotion_match = re.search(emotion_pattern, result, re.IGNORECASE)
                emotion = int(emotion_match.group(1)) if emotion_match else None
                emotion = max(1, min(5, emotion))

                behavior_match = re.search(behavior_pattern, result, re.IGNORECASE)
                behavior = behavior_match.group("behavior").strip().lower() if behavior_match else None

                stage_match = re.search(stage_pattern, result, re.IGNORECASE)
                stage = stage_match.group("stage").strip().lower() if stage_match else None

                print(f"Extracted - Emotion: {emotion}, Behavior: {behavior}, Stage: {stage}")
                if not all([emotion, behavior, stage]):
                    raise ValueError("Some fields missing")

            except Exception as e:
                print(f"[Fallback triggered due to]: {e}")

                def extract_last_text(matches):
                    if not matches:
                        return None
                    last = matches[-1]
                    if isinstance(last, tuple):
                        return last[-1].strip()
                    return last.strip()
                print(output_str)
                emo_lst = re.findall(emotion_pattern, output_str, re.IGNORECASE)
                print(emo_lst,flush=True)
                emotion = int(extract_last_text(emo_lst)) if emo_lst else 3
                behavior_lst = re.findall(behavior_pattern, output_str, re.IGNORECASE)
                print(behavior_lst,flush=True)
                behavior = extract_last_text(behavior_lst) or "others"
                stage_lst = re.findall(stage_pattern, output_str, re.IGNORECASE)
                print(stage_lst,flush=True)
                stage = extract_last_text(stage_lst) or "precontemplation"

            state.update({
                "emotion": emotion,
                "behavior": behavior.lower(),
                "stage": stage.lower()
            })

        elif role=="state_tracking_trust":
            output = output.lower()
            print("--trust--\n",flush=True)
            print(output,flush=True)
            # 定义正则 pattern，支持 ":" 或 "="，允许大小写混写
            reliability_pattern = r"reliability\s*[:=]\s*(\d+)"
            response_competence_pattern = r"response competence\s*[:=]\s*(\d+)"
            perceived_understandability_pattern = r"perceived understandability\s*[:=]\s*(\d+)"
            output_parts = output.split("\n\n")
            try:
                result = output_parts[-1]
                reliability_match = re.search(reliability_pattern, result, re.IGNORECASE)
                response_competence_match = re.search(response_competence_pattern, result, re.IGNORECASE)
                perceived_understandability_match = re.search(perceived_understandability_pattern, result,
                                                              re.IGNORECASE)
                if reliability_match and response_competence_match and perceived_understandability_match:
                    reliability = int(reliability_match.group(1))
                    response_competence = int(response_competence_match.group(1))
                    perceived_understandability = int(perceived_understandability_match.group(1))
                else:
                    # fallback: 直接用数值提取（容忍字段名拼写错误）
                    scores = re.findall(r'[:=]\s*([1-5])\b', result)
                    if len(scores) >= 3:
                        reliability = int(scores[-3])
                        response_competence = int(scores[-2])
                        perceived_understandability = int(scores[-1])
                    else:
                        raise ValueError(f"Not enough scores found: {scores}")
            except Exception as e:
                print(f"[Primary match failed: {e}]",flush=True)
                print(output_parts,flush=True)
                # fallback: 全文搜索数值
                scores = re.findall(r'[:=]\s*([1-5])\b', output)
                if len(scores) >= 3:
                    reliability = int(scores[-3])
                    response_competence = int(scores[-2])
                    perceived_understandability = int(scores[-1])
                else:
                    reliability = 3
                    response_competence = 3
                    perceived_understandability = 3

                print(
                    f"[Fallback] reliability: {reliability}, response_competence: {response_competence}, perceived_understandability: {perceived_understandability}")

            trust = average_to_5_level(
                np.mean([reliability, response_competence, perceived_understandability])
            )

            state.update({
                "trust": trust
            })
        else:
            emotion, behavior, stage, reliability, response_competence, perceived_understandability = re.split(r',',
                                                                                                           result[-1])
            trust = average_to_5_level(
                np.mean([int(reliability), int(response_competence), int(perceived_understandability)]))
            state.update( {
                "emotion": int(emotion),
                "behavior": behavior.strip().lower(),
                "stage": stage.strip().lower(),
                "trust": trust
            })
        return state

    def get_critic(self, model_name, messages, model):
        print("Get Critic")
        if model == 'llama2':
            prompt = llama2_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            with torch.no_grad():
                output_ids = self.vicuna_model.generate(
                    torch.as_tensor(input_ids).to(self.vicuna_device),
                    max_new_tokens=16,
                    temperature = 1.1,
                    do_sample = True,
                    early_stopping=True,
                    num_return_sequences=3
                )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'qwen3':
            prompt = self.vicuna_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            with torch.no_grad():
                output_ids = self.vicuna_model.generate(
                    torch.as_tensor(input_ids).to(self.vicuna_device),
                    max_new_tokens=16,
                    temperature=1.1,
                    do_sample=True,
                    early_stopping=True,
                    num_return_sequences=3,
                )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'chatgpt':
            messages = chatgpt_prompt(messages, user_role[self.args.data_name])
            outputs = query_chat_ai_model(
                api_key=YOUR_API_KEY,
                messages=messages,
                model="gpt-3.5-turbo-0613",
                max_tokens=self.args.max_new_tokens,
                temperature=1.1,
                n=3
            )
        elif model == 'chat_ai':
            messages = chatgpt_prompt(messages, user_role[self.args.data_name])
            outputs = query_chat_ai_model(
                messages=messages,
                model=model_name,
                max_tokens=16,
                temperature=1.1,
                n=3
            )
        if self.args.data_name in ['esc']:
            rewards = []
            for output in outputs:
                for key in self.reward_dict[self.args.data_name]:
                    if key in output.lower():
                        rewards.append(self.reward_dict[self.args.data_name][key])
                        break
            if len(rewards) == 0:
                reward = 0
            else:
                reward = sum(rewards)/len(rewards)
        print(outputs,flush=True)
        return reward

    def compute_reward(self, emotion, trust, critic_score=0):
        """
        计算奖励（平衡版）：
        - 每 N 轮计算一次
        - 情绪变化 → 0.3 / 信任变化 → 0.3 / critic_score → 0.4
        """
        reward = 0
        if len(self.past_emotions) == self.N:  # 仅在累积 N 轮后计算
            idx = random.randint(0, len(self.past_emotions) - 1)
            # idx=0
            past_emotion = self.past_emotions[idx]
            past_trust = self.past_trusts[idx]

            # 计算变化量
            delta_emotion = past_emotion - emotion  # 下降为正
            delta_trust = trust - past_trust  # 上升为正

            # 情绪奖励（权重 0.3）
            if delta_emotion > 0:
                reward += delta_emotion * 0.3
            elif delta_emotion < 0:
                reward += delta_emotion * 0.3

            # 信任奖励（权重 0.3）
            if delta_trust > 0:
                reward += delta_trust * 0.3
            elif delta_trust < 0:
                reward += delta_trust * 0.3
            # 清除旧数据
            self.past_emotions.pop(0)
            self.past_trusts.pop(0)

        # 记录新数据
        self.past_emotions.append(emotion)
        self.past_trusts.append(trust)
        return reward + 0.4 * critic_score