
from torch.distributions import Categorical
from torch.optim import AdamW
from transformers import BertModel, RobertaModel, AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from prompt import ESConvAct,behaviors,stages#, CIMAAct, CBAct
import math
PROJECT_DIR = "/projects/scc/UGOE/UXEI/UMIN/scc_umin_ag_xiaoming_fu/umin_kurs_datascismartcity_s26/dir.project/lj/LLM"
model = {'bert': BertModel, 'roberta': RobertaModel}
act = {'esc': ESConvAct}# 'cima': CIMAAct, 'cb': CBAct
TMP_DIR = {
    'esc': PROJECT_DIR+'Dialogue_Planning_LLM/tmp/esc',
    'cima': './tmp/cima',
    'cb': './tmp/cb',
}
class AttentionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states, mask=None):
        # hidden_states: [batch_size, seq_len, hidden_size]
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_probs = self.softmax(attn_weights)

        output = torch.matmul(attn_probs, V)
        pooled = output.mean(dim=1)  # 可替换为 CLS or attention weighted pool
        return pooled
class MLPFeatureExtractor(nn.Module):
    """ 用于提取不同维度（情绪、信任度、行为、改变阶段）的特征 """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPFeatureExtractor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)  # (batch_size, seq_len, output_dim)

class MDDP(nn.Module):
    def __init__(self, args, config, tokenizer):
        super().__init__()
        self.policy = model[args.model_name].from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir)
        self.hidden_size=config.hidden_size
        if args.state_attention==True:
            self.emotion_embedding=nn.Embedding(5, config.hidden_size)
            self.behavior_embedding = nn.Embedding(9, config.hidden_size)
            self.stage_embedding = nn.Embedding(5, config.hidden_size)
            self.trust_embedding = nn.Embedding(5, config.hidden_size)
            self.W1 = nn.Linear(config.hidden_size, config.hidden_size,bias=False)
            self.W2 = nn.Linear(config.hidden_size, config.hidden_size,bias=False)
            self.W3 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size,config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size,config.hidden_size)#4
            )
            self.layernorm_state = nn.LayerNorm(config.hidden_size)
            self.convert_ffn = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
            self.value_function =  nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1)
        )
        self.act = sorted(list(act[args.data_name].keys()))
        self.classifier = nn.Linear(config.hidden_size, len(self.act))
        self.tokenizer = tokenizer
        self.optimizer = AdamW(
            self.parameters(), lr=args.dp_learning_rate
        )
        self.args = args
        self.rewards = []
        self.old_log_probs = []      # 采样时旧策略的 log π_old(a|s)，detach
        self.saved_values = []       # 采样时旧策略的 value 估计
        self.action_indices = []     # 每步采样的动作索引
        self.saved_states = []       # 保存 forward 所需输入，优化阶段用当前参数重新前向

    def build_input(self, state):
        dial_id = []
        for turn in state[::-1]:
            s = self.tokenizer.encode("%s: %s" % (turn['role'], turn['content']))
            if len(dial_id) + len(s) > self.args.max_seq_length:
                break
            dial_id = s[1:] + dial_id
        inp = s[:1] + dial_id
        return [inp]

    def forward(self, input_ids,attention_mask,emotion,trust,stage,behavior,train_mode='supervised'):
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0]  # (B, H)

        if self.args.state_attention==True:
        # new
            emo_state = self.emotion_embedding(emotion)
            stage_state = self.stage_embedding(stage)
            behavior_state = self.behavior_embedding(behavior)
            trust_state = self.trust_embedding(trust)
            state_embed = torch.stack([emo_state, trust_state, stage_state, behavior_state], dim=1)
            state_embed= self.mlp(state_embed)
            state_features = self.layernorm_state(state_embed)
            d_k=state_features.size(-1)
            s_attn_scores = torch.matmul(self.W1(cls_output).unsqueeze(1),
                                  self.W2(state_features).transpose(-2, -1)) / math.sqrt(d_k)  # (B, Lc, Lctx)
            s_attn_weights = torch.softmax(s_attn_scores, dim=-1)  # (B, Lc, Lctx)

            state_features=torch.matmul(s_attn_weights, self.W3(state_features)) # (B, Lc, d)
            final_output = torch.cat([state_features.squeeze(1), cls_output], dim=-1)

            strategy_feature = self.convert_ffn(final_output)
            logits = self.classifier(strategy_feature)
        else:
            strategy_feature=cls_output
            logits = self.classifier(strategy_feature)
        if train_mode == 'supervised':
            return logits,strategy_feature
        elif train_mode == 'rl':
            value = self.value_function(strategy_feature)
            return logits, strategy_feature,value
        elif train_mode == 'test':
            if  self.args.state_attention==True:
                state_attention = self.analyze_state_attention(
                    s_attn_weights=s_attn_weights
                )
                feature_importance = self.analyze_feature_importance()
            else:
                state_attention=None
                feature_importance=None
            return logits, strategy_feature,state_attention,feature_importance

    def select_action(self, state,emotion,trust,stage,behavior, is_test=False,train_mode='supervised'):
        """
        选择动作的方法
        保存动作索引用于精确的策略比率计算
        """
        if self.args.dp_device:
            device = self.args.dp_device
        else:
            device=self.args.device
        if self.args.state_attention==True:
            trust = torch.tensor([trust], dtype=torch.long, device=device)
            emotion = torch.tensor([emotion], dtype=torch.long, device=device)
            behavior = torch.tensor([behavior], dtype=torch.long, device=device)
            stage = torch.tensor([stage], dtype=torch.long, device=device)
        inp = self.build_input(state)
        inp = torch.tensor(inp,dtype=torch.long, device=device)
        attention_mask = inp.ne(0)
        if train_mode == 'supervised':
            logits,states = self.forward(inp, attention_mask, emotion,trust,stage,behavior,train_mode='supervised')
            probs = nn.functional.softmax(logits, dim=-1)
            m = Categorical(probs)

        elif train_mode == 'rl':
            logits, states,value = self.forward(inp, attention_mask,emotion,trust,stage,behavior, train_mode='rl')
            probs = nn.functional.softmax(logits, dim=-1)
            m = Categorical(probs)

        if is_test:
            action = probs.argmax().item()
            return self.act[action]
        else:
            action = m.sample()
            action_idx = action.item()

            # ==================== PPO: 保存采样信息（旧策略） ====================
            self.saved_values.append(value.detach())
            self.old_log_probs.append(m.log_prob(action).detach())
            self.action_indices.append(action_idx)
            self.saved_states.append({
                'input_ids': inp.detach(),
                'attention_mask': attention_mask.detach(),
                'emotion': emotion.detach() if self.args.state_attention else emotion,
                'trust': trust.detach() if self.args.state_attention else trust,
                'stage': stage.detach() if self.args.state_attention else stage,
                'behavior': behavior.detach() if self.args.state_attention else behavior,
            })
            # ==================== 修正结束 ====================

            return self.act[action_idx],states,value


    def compute_advantages_gae(self, rewards, values, gamma=0.99, gae_lambda=0.95):
        """
        计算广义优势估计（GAE）
        参数:
            rewards: 奖励序列 [T]
            values: 价值函数估计序列 [T]
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
        返回:
            advantages: 优势函数序列 [T]
        """
        advantages = []
        gae = 0

        # 确保转换为张量（rewards 可能是 tensor 列表）
        if isinstance(rewards, list) and len(rewards) > 0 and torch.is_tensor(rewards[0]):
            rewards_tensor = torch.stack(rewards).reshape(-1).to(dtype=torch.float32, device=values.device)
        else:
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=values.device)

        # ==================== next_values（V(s_{t+1})） ====================
        T = len(rewards)
        if len(values) == T + 1:        # values 含终止状态 bootstrap
            next_values = values[1:]
            values = values[:-1]
        elif len(values) == T:          # 每步一个 value，末步 bootstrap 用 0
            next_values = torch.cat([values[1:], values.new_zeros(1)])
        else:
            next_values = values

        # 反向计算 GAE：delta = r_t + gamma * V(s_{t+1}) - V(s_t)
        for t in reversed(range(T)):
            delta = rewards_tensor[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)

        return torch.stack(advantages) if len(advantages) > 0 else torch.zeros(0, device=values.device)

    def optimize_model_corrected(self, train_mode='rl'):
        """
        PPO优化方法：
        - 使用returns-old_values计算advantage（与value目标同尺度）
        - 添加entropy正则化
        - 保留k-epoch多次更新机制
        """
        device = self.args.dp_device

        if len(self.rewards) == 0:
            print("No rewards found. Skipping optimization.")
            return 0.0

        print("Rewards:", self.rewards)
        rewards = torch.stack(self.rewards).reshape(-1).to(device=device, dtype=torch.float32)

        if len(self.saved_values) == 0:
            print("No saved values found. Skipping optimization.")
            return 0.0

        old_values = torch.stack(self.saved_values).squeeze().detach()  # [T]
        old_log_probs = torch.stack(self.old_log_probs).detach()  # [T]  ← 修正这里！

        if len(self.saved_states) != len(self.rewards):
            print("Warning: saved_states length mismatch with rewards. Skipping optimization.")
            return 0.0

        returns = torch.zeros_like(rewards, device=device)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.args.gamma * R
            returns[t] = R

        # 标准化returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # advantage = returns - old_values
        advantages = returns - old_values

        # 标准化advantage
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 参数（保留k-epoch）
        clip_epsilon = getattr(self.args, 'eps_clip', 0.2)
        value_coeff = getattr(self.args, 'value_coeff', 0.5)
        entropy_coeff = getattr(self.args, 'entropy_coeff', 0.00)
        ppo_epochs = getattr(self.args, 'ppo_epochs', 1)
        max_grad_norm = getattr(self.args, 'max_grad_norm', 0.5)

        last_total_loss = 0.0

        # k-epoch 多次更新
        for epoch in range(ppo_epochs):
            policy_losses = []
            value_losses = []
            entropy_terms = []

            for i, st in enumerate(self.saved_states):
                # 用当前参数重新前向
                logits, _, value = self.forward(
                    st['input_ids'], st['attention_mask'],
                    st['emotion'], st['trust'], st['stage'], st['behavior'],
                    train_mode='rl'
                )
                dist = Categorical(nn.functional.softmax(logits, dim=-1))
                a = torch.tensor(self.action_indices[i], device=logits.device)

                new_log_prob = dist.log_prob(a)
                entropy = dist.entropy()

                ratio = torch.exp(new_log_prob - old_log_probs[i])
                clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

                adv = advantages[i]
                policy_loss = -torch.min(ratio * adv, clipped_ratio * adv)

                value = value.reshape(-1).mean()
                value_loss = F.mse_loss(value, returns[i])

                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropy_terms.append(entropy)

            if len(policy_losses) == 0:
                return 0.0

            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy_loss = torch.stack(entropy_terms).mean()

            total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy_loss

            print(f"Epoch {epoch+1}/{ppo_epochs}: Value loss: {value_loss.item():.6f}, "
                  f"Policy loss: {policy_loss.item():.6f}, Entropy: {entropy_loss.item():.6f}",
                  flush=True)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            last_total_loss = total_loss.item()

        if torch.cuda.is_available():
            cached = torch.cuda.memory_reserved(device) / 1024 / 1024
            print(f"GPU memory after optimization: {cached:.2f} MB", flush=True)

        self.clear_buffer_enhanced()
        return last_total_loss

    def clear_buffer_enhanced(self):
        """增强的缓冲区清理方法"""
        self.rewards.clear()
        self.old_log_probs.clear()
        self.saved_values.clear()
        self.action_indices.clear()
        self.saved_states.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保持原始方法作为备选，但标记为有问题的版本
    def optimize_model(self, train_mode='rl'):
        return self.optimize_model_corrected(train_mode)

    def save_model(self, data_name, filename, epoch_user):
            output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
            torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))

    def load_model(self, data_name, filename, epoch_user=None):
        if epoch_user:
            output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        else:
            output_dir = filename
        print(output_dir,flush=True)
        if hasattr(self, 'module'):
            self.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model-cat-13-new.bin')),strict=False)
        else:
            self.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model-cat-13-new.bin'), map_location=self.args.dp_device),strict=False)

    def reset(self):
        """重置缓冲区"""
        self.rewards.clear()
        self.old_log_probs.clear()
        self.saved_values.clear()
        self.action_indices.clear()
        self.saved_states.clear()

    def analyze_state_attention(self, s_attn_weights):

        """
        计算四种状态（情绪、信任度、行为、改变阶段）的关注度分布。
        返回一个字典，包含每种状态的注意力权重。
        """
        # 提取四种状态的注意力权重
        attention_distribution = {
            'emotion': s_attn_weights[:,:, 0].item(),
            'trust': s_attn_weights[:,:, 1].item(),
            'stage': s_attn_weights[:,:, 2].item(),
            'behavior': s_attn_weights[:,:, 3].item()
        }
        return attention_distribution


    def analyze_feature_importance(self):
        """
        计算 state_features 和 cls_output 的关注度。
        返回一个字典，包含两者的注意力权重。
        """
        # 计算注意力权重
        with torch.no_grad():
            # 获取 convert_ffn 的第一层权重
            weights = self.convert_ffn[0].weight  # (H, H*2)
            # bias = self.convert_ffn[0].bias  # (H,)

            # 分割权重，分别对应 state_features 和 cls_output
            state_weights = weights[:, :self.hidden_size]  # (H, H)
            cls_weights = weights[:, self.hidden_size:]  # (H, H)

            # 计算权重范数（L2范数）作为关注度指标
            state_norm = torch.norm(state_weights, p=2, dim=1).mean().item()
            cls_norm = torch.norm(cls_weights, p=2, dim=1).mean().item()

            # 归一化为百分比
            total_norm = state_norm + cls_norm
            state_attention = state_norm / total_norm * 100
            cls_attention = cls_norm / total_norm * 100

        return {
            'state_features_attention': state_attention,
            'cls_output_attention': cls_attention
        }
class BridgeBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.65):
        super(BridgeBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.kg_cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ins_cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                                batch_first=True)
        self.feed_forward1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward_query1=nn.Linear(hidden_dim,hidden_dim)
        self.feed_forward2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.feed_forward_query2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm_f1 = nn.LayerNorm(hidden_dim)
        self.norm_f2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self,context,knowledge,instruction,knowledge_query,instruction_query,knowledge_mask,instruction_mask,self_attn_mask):
        if knowledge is not None:
            query1_len = instruction_query.size(1)
            query2_len = knowledge_query.size(1)
            # Self-attention
            x = torch.cat([instruction_query,knowledge_query, context], dim=1)
            self_attn_output, _ = self.self_attn(x, x, x,key_padding_mask=self_attn_mask)
            fused_ins = self_attn_output[:, :query1_len, :]
            fused_kg = self_attn_output[:, query1_len:query1_len+query2_len, :]
            # knowledge
            # Cross-attention with knowledge

            cross_attn_output1, _ = self.kg_cross_attn(fused_kg, knowledge, knowledge,key_padding_mask=knowledge_mask)
            x_kg = self.norm1(fused_kg + self.dropout(cross_attn_output1))
            # Feed-forward
            x1 = self.feed_forward1(x_kg)
            x_kg = self.norm_f1(x_kg + self.dropout(x1))
            kg_query = self.feed_forward_query1(x_kg)
            #instruction
            cross_attn_output2, _ = self.ins_cross_attn(fused_ins, instruction, instruction,key_padding_mask=instruction_mask)
            x_ins = self.norm2(fused_ins + self.dropout(cross_attn_output2))
            x2 = self.feed_forward2(x_ins)
            x_ins = self.norm_f2(x_ins + self.dropout(x2))
            ins_query = self.feed_forward_query2(x_ins)
        else:
            query1_len = instruction_query.size(1)
            # Self-attention
            x = torch.cat([instruction_query,context], dim=1)
            self_attn_output, _ = self.self_attn(x, x, x,key_padding_mask=self_attn_mask)
            fused_ins = self_attn_output[:, :query1_len, :]
            # instruction
            cross_attn_output2, _ = self.ins_cross_attn(fused_ins, instruction, instruction)
            x_ins = self.norm2(fused_ins + self.dropout(cross_attn_output2))
            x2 = self.feed_forward2(x_ins)
            x_ins = self.norm_f2(x_ins + self.dropout(x2))
            ins_query = self.feed_forward_query2(x_ins)
            kg_query=None
        # ff_history = self.feed_forward_history(cross_attn_output[:, query_len:, :])
        return kg_query, ins_query

class PG(nn.Module):
    def __init__(self, args, config, tokenizer):
        super().__init__()
        self.kg_layers = nn.ModuleList([
            BridgeBlock(config.hidden_size, config.num_attention_heads, 0.5)
            for _ in range(1)
        ])
        self.tokenizer = tokenizer
        self.args = args
        self.query1 = nn.Parameter(torch.randn(1, 8, config.hidden_size))
        self.query2 = nn.Parameter(torch.randn(1,64, config.hidden_size))
        self.log_std = nn.Parameter(torch.zeros(config.hidden_size))
        self.value_function = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, 1)
        )
        self.eps = np.finfo(np.float32).eps.item()
        self.optimizer = AdamW(self.parameters(), lr=args.pg_learning_rate)
        self.config = config
        self.args = args
        self.rewards = []
        self.old_log_probs = []
        self.saved_values = []
        self.saved_actions = []
        self.saved_inputs = []

    def _rl_forward_dist(self, context_embeddings, attention_mask, knowledge_embeddings,
                         instruction_embeddings, knowledge_mask=None, instruction_mask=None):
        """
        重新执行 rl 前向到分布与 value（不采样），用于 PPO 优化阶段计算新策略 log_prob。
        返回: dist (Normal over soft_prompt_original), value [B]
        """
        bsz = context_embeddings.size()[0]
        instruction_query = self.query1.repeat(bsz, 1, 1)
        if self.args.RAG == True:
            knowledge_query = self.query2.repeat(bsz, 1, 1)
        else:
            knowledge_query = None
            knowledge_embeddings = None

        if attention_mask is not None:
            q_len = instruction_query.size(1) + (knowledge_query.size(1) if knowledge_query is not None else 0)
            query_mask = attention_mask.new_zeros(bsz, q_len)  # query 段永不屏蔽
            ctx_pad_mask = ~attention_mask  # True = context 的 pad 位置
            self_attn_mask = torch.cat([query_mask, ctx_pad_mask], dim=1)  # [B, q_len + L_ctx]
        else:
            self_attn_mask = None

        for layer in self.kg_layers:
            knowledge_query, instruction_query = layer(
                context_embeddings, knowledge_embeddings, instruction_embeddings,
                knowledge_query, instruction_query, knowledge_mask, instruction_mask,self_attn_mask)

        if self.args.RAG == True:
            soft_prompt_original = torch.cat([instruction_query, knowledge_query], dim=1)
        else:
            soft_prompt_original = instruction_query
        log_std = torch.clamp(self.log_std, -5, 2)
        std = torch.exp(log_std).unsqueeze(0).unsqueeze(0).expand(bsz, soft_prompt_original.size(1), -1)
        dist = torch.distributions.Normal(soft_prompt_original, std)
        instruction_pooled = instruction_query.mean(dim=1)
        knowledge_pooled = knowledge_query.mean(dim=1)
        value = self.value_function(torch.cat([instruction_pooled, knowledge_pooled], dim=-1)).squeeze(-1)
        return dist, value

    def reset(self):
        self.rewards.clear()
        self.old_log_probs.clear()
        self.saved_values.clear()
        self.saved_actions.clear()
        self.saved_inputs.clear()

    def build_input(self, external_knowledge, instruction, dialog_context):
        """
        分别构建外部知识、指令和对话上下文的输入。
        """
        context_ids = self.tokenizer.encode(dialog_context, return_tensors='pt')
        knowledge_ids = self.tokenizer.encode(external_knowledge, return_tensors='pt')
        instruction_ids = self.tokenizer.encode(instruction, return_tensors='pt')
        return context_ids, knowledge_ids, instruction_ids

    def select_knowledge(self, policy, external_knowledge):
        """
        根据策略 policy 选择合适的外部知识。
        """
        # 这里可以使用检索机制或打分机制选择合适的知识
        selected_knowledge = external_knowledge.get(policy, "")
        return selected_knowledge

    def forward(self,context_embeddings,attention_mask, knowledge_embeddings,instruction_embeddings,knowledge_mask=None,instruction_mask=None,train_mode="sft"):
        """
        前向传播过程，先通过 Q-Former 处理输入，再输入到 base model。
        """
        bsz=context_embeddings.size()[0]
        instruction_query = self.query1.repeat(bsz, 1, 1)
        if self.args.RAG==True:
            knowledge_query= self.query2.repeat(bsz, 1, 1)
        else:
            knowledge_query=None
            knowledge_embeddings = None

        if attention_mask is not None:
            q_len = instruction_query.size(1) + (knowledge_query.size(1) if knowledge_query is not None else 0)
            query_mask = attention_mask.new_zeros(bsz, q_len)
            ctx_pad_mask = ~attention_mask  # True = context 的 pad 位置
            self_attn_mask = torch.cat([query_mask, ctx_pad_mask], dim=1)  # [B, q_len + L_ctx]
        else:
            self_attn_mask = None

        for layer in self.kg_layers:
            knowledge_query,instruction_query = layer(context_embeddings, knowledge_embeddings,instruction_embeddings,knowledge_query,instruction_query,knowledge_mask,instruction_mask,self_attn_mask)
        if self.args.RAG==True:
            soft_prompt=torch.cat([instruction_query,knowledge_query,context_embeddings],dim=1)
        else:
            soft_prompt = torch.cat([ instruction_query,context_embeddings], dim=1)
        if train_mode=="rl":
            # Project each token independently
            if self.args.RAG==True:
                soft_prompt_original = torch.cat([instruction_query, knowledge_query], dim=1)
            else:
                soft_prompt_original = instruction_query
            L_total = soft_prompt_original.size(1)
            # --- Gaussian policy for sequence ---
            # std: [1, 1, H] -> broadcast to [B, L, H]
            log_std = torch.clamp(self.log_std, -5, 2)
            std = torch.exp(log_std).unsqueeze(0).unsqueeze(0)  # [1, 1, H]
            std = std.expand(bsz, L_total, -1)  # [B, L, H]
            dist = torch.distributions.Normal(soft_prompt_original, std)
            soft_prompt_re = dist.rsample()  # [B, L, H]

            log_prob = dist.log_prob(soft_prompt_re).mean(dim=[1, 2])  # [B], mean over tokens & dim
            # prob = torch.exp(log_prob)
            # --- Value function ---
            # 平均池化后估计 value（也可以用 CLS 或 attention pooling）
            instruction_pooled = instruction_query.mean(dim=1)
            knowledge_pooled = knowledge_query.mean(dim=1)# [B, H]
            value = self.value_function(torch.cat([instruction_pooled, knowledge_pooled], dim=-1)).squeeze(-1)  # [B]
            # Save for PPO
            self.saved_values.append(value.detach())
            self.old_log_probs.append(log_prob.detach())
            # ==================== 保存采样动作与前向输入，供优化阶段用当前参数重算 ====================
            self.saved_actions.append(soft_prompt_re.detach())
            self.saved_inputs.append({
                'context_embeddings': context_embeddings.detach(),
                'attention_mask': attention_mask.detach() if attention_mask is not None else None,
                'knowledge_embeddings': knowledge_embeddings.detach() if knowledge_embeddings is not None else None,
                'instruction_embeddings': instruction_embeddings.detach() if instruction_embeddings is not None else None,
                'knowledge_mask': knowledge_mask.detach() if knowledge_mask is not None else None,
                'instruction_mask': instruction_mask.detach() if instruction_mask is not None else None,
            })
            soft_prompt=torch.cat([soft_prompt_re,context_embeddings],dim=1)
            return soft_prompt, log_prob, value
        else:
            return soft_prompt

    def compute_advantages_gae_pg(self, rewards, values, gamma=0.99, gae_lambda=0.95):
        """
        PG类的GAE计算方法
        """
        advantages = []
        gae = 0

        # 确保转换为张量（rewards 可能是 tensor 列表）
        if isinstance(rewards, list) and len(rewards) > 0 and torch.is_tensor(rewards[0]):
            rewards_tensor = torch.stack(rewards).reshape(-1).to(dtype=torch.float32, device=values.device)
        else:
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=values.device)

        # ==================== 正确构造 next_values（V(s_{t+1})） ====================
        T = len(rewards)
        if len(values) == T + 1:
            next_values = values[1:]
            values = values[:-1]
        elif len(values) == T:
            next_values = torch.cat([values[1:], values.new_zeros(1)])
        else:
            next_values = values

        # 反向计算GAE：delta = r_t + gamma * V(s_{t+1}) - V(s_t)
        for t in reversed(range(T)):
            delta = rewards_tensor[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)

        return torch.stack(advantages) if len(advantages) > 0 else torch.zeros(0, device=values.device)

    def optimize_PPO_corrected(self):
        """
        PG类PPO优化方法
        """
        if len(self.rewards) == 0:
            return 0.0
        if self.args.pg_device:
            device = self.args.pg_device
        else:
            device = self.args.device
        if len(self.saved_values) == 0 or len(self.old_log_probs) == 0:
            return 0.0
        if len(self.saved_inputs) != len(self.rewards):
            print("Warning: PG saved_inputs length mismatch with rewards. Skipping optimization.")
            return 0.0

        old_values = torch.stack(self.saved_values).reshape(-1).detach()       # [T]
        old_log_probs = torch.stack(self.old_log_probs).reshape(-1).detach()   # [T]

        # 获取模型dtype（parameters可能在bfloat16），确保returns/advantages与之对齐
        model_dtype = next(self.parameters()).dtype

        # ==================== 统一用returns计算advantage和value目标 ====================
        rewards_tensor = torch.stack(self.rewards).reshape(-1).to(device)

        # 蒙特卡洛回报
        returns = torch.zeros_like(rewards_tensor)
        R = 0
        for t in reversed(range(len(rewards_tensor))):
            R = rewards_tensor[t] + self.args.gamma * R
            returns[t] = R

        # 标准化returns，并cast到model_dtype
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = returns.to(dtype=model_dtype)

        # 修正：advantage = returns - old_values（TD目标，与value同尺度）
        advantages = returns - old_values

        # 标准化advantage
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ==================== PPO 参数（保留k-epoch）====================
        clip_epsilon = getattr(self.args, 'eps_clip', 0.2)
        value_coeff = getattr(self.args, 'value_coeff', 0.5)
        entropy_coeff = getattr(self.args, 'entropy_coeff', 0.01)  # 修正：默认0.01
        ppo_epochs = getattr(self.args, 'ppo_epochs', 1)  # k-epoch，默认1
        max_grad_norm = getattr(self.args, 'max_grad_norm', 0.5)

        last_total_loss = 0.0

        # ==================== k-epoch 多次更新 ====================
        for epoch in range(ppo_epochs):
            policy_losses = []
            value_losses = []
            entropy_terms = []

            for i, inp in enumerate(self.saved_inputs):
                # 用当前参数重新前向得到新分布与value
                dist, value = self._rl_forward_dist(
                    inp['context_embeddings'], inp['attention_mask'],
                    inp['knowledge_embeddings'], inp['instruction_embeddings'],
                    inp['knowledge_mask'], inp['instruction_mask']
                )
                action = self.saved_actions[i]  # [B, L, H]

                # 保持[B]维度，与old_log_probs一致，不要额外mean()
                new_log_prob = dist.log_prob(action).mean(dim=[1, 2])  # [B]
                entropy = dist.entropy().mean(dim=[1, 2])  # [B]

                # ratio在[B]维度计算，确保与old_log_probs[i]同维度
                ratio = torch.exp(new_log_prob - old_log_probs[i])  # [B]

                # 对每个样本分别clip
                clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                adv = advantages[i]

                # 先对每个元素计算min，然后mean得到标量
                policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

                value = value.reshape(-1).mean()
                value_loss = F.mse_loss(value, returns[i])

                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropy_terms.append(entropy.mean())

            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy_loss = torch.stack(entropy_terms).mean()

            # entropy_coeff非0，鼓励探索
            # 强制所有loss项和total_loss统一为model_dtype，避免backward时dtype不匹配
            total_loss = (policy_loss.to(dtype=model_dtype)
                        + value_coeff * value_loss.to(dtype=model_dtype)
                        - entropy_coeff * entropy_loss.to(dtype=model_dtype))

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            last_total_loss = total_loss.item()

            print(f"Epoch {epoch+1}/{ppo_epochs}: Value loss: {value_loss.item():.6f}, "
                  f"Policy loss: {policy_loss.item():.6f}, Entropy: {entropy_loss.item():.6f}",
                  flush=True)

        if torch.cuda.is_available():
            cached = torch.cuda.memory_reserved(device) / 1024 / 1024
            print(f"PG GPU memory after optimization: {cached:.2f} MB", flush=True)

        self.clear_buffer_enhanced_pg()
        return last_total_loss

    def clear_buffer_enhanced_pg(self):
        """PG类增强的缓冲区清理"""
        self.rewards.clear()
        self.old_log_probs.clear()
        self.saved_values.clear()
        self.saved_actions.clear()
        self.saved_inputs.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 保持原始方法但标记为有问题的版本
    def optimize_PPO(self):
        """
        原始PG类PPO优化方法（建议使用optimize_PPO_corrected）
        """
        # print("WARNING: Using original PG optimize_PPO which can be improved.")
        # print("Recommend using optimize_PPO_corrected for better stability.")
        return self.optimize_PPO_corrected()
    # ==================== 修正结束 ====================
    def optimize_model(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        if rewards.shape[0] > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
        return policy_loss.data



    def save_model(self, data_name, filename, epoch_user):
        output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))

    def load_model(self, data_name, filename, epoch_user=None):
        if epoch_user:
            output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        else:
            output_dir = filename
        if hasattr(self, 'module'):
            self.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model-64-42.bin')))
        else:
            self.load_state_dict(
                torch.load(os.path.join(output_dir, 'pytorch_model-64-42.bin'), map_location=self.args.pg_device))#-ep5-13
#pytorch_model2.bin
