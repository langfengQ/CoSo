import torch
from transformers import AutoTokenizer
from digirl.models.critic import VLMDoubleCritic, TrajectoryCritic
from digirl.models.causal import MiniBertConfig, MiniBertForSequenceClassification
from digirl.environment.android.autoui_utils import ActionType
from digirl.coso import CausalExecutor, parse_action
from .model import T5ForMultimodalGeneration
import signal

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

class AutoUIAgent(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm = "gpt2", critic_lm = "roberta-base", 
                cache_dir = '~/.cache', dropout = 0.5, TEMPLATE = None, use_lora=False,
                do_sample = True, temperature = 1.0, max_new_tokens = 32, use_bfloat16 = False, eos_str = None,
                use_causal = False, causal_save_path = None,
                ):
        super(AutoUIAgent, self).__init__()
        if use_bfloat16:
            self.model = T5ForMultimodalGeneration.from_pretrained(policy_lm, cache_dir=cache_dir,
                                                              torch_dtype = torch.bfloat16).to(device)
        else:
            self.model = T5ForMultimodalGeneration.from_pretrained(policy_lm, cache_dir=cache_dir).to(device)
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            lora_config = LoraConfig(
                r=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=32,
                lora_dropout=0.05
            )
            self.model = get_peft_model(self.model, lora_config)
            print("Using LoRA")
            self.model.print_trainable_parameters()
        self.template = TEMPLATE
        self.policy_lm = policy_lm
        self.critic = VLMDoubleCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = 768, out_dim = 1)  
        self.trajectory_critic = TrajectoryCritic(device, accelerator, critic_lm = critic_lm, cache_dir = cache_dir, in_dim = 768, out_dim = 1)
        self.target_critic = None
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        config = MiniBertConfig(vocab_size=self.tokenizer.vocab_size, 
                                hidden_size=128, 
                                intermediate_size = 128 * 4,
                                num_labels=len(ActionType), 
                                num_attention_heads=4, 
                                num_hidden_layers=3,
                                )
        self.use_causal = use_causal
        if self.use_causal:
            self.causal_model = MiniBertForSequenceClassification(config, accelerator).to(device)
            self.causal_executor = CausalExecutor(self.tokenizer, device, causal_save_path = causal_save_path)

        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim= -1)
        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
        self.eos_str = eos_str
    
    def prepare(self):
        self.model = self.accelerator.prepare(self.model)
        self.critic = self.accelerator.prepare(self.critic)
        self.trajectory_critic = self.accelerator.prepare(self.trajectory_critic)
        if self.use_causal:
            self.causal_model = self.accelerator.prepare(self.causal_model)

    def get_action(self, observation, image_features):
        image_features = image_features[..., -1408:]
        # if self.template is not None:
        #     observation = [self.template.replace("{obs}", obs) for obs in observation]
        for _ in range(3):
            try:
                with timeout(seconds=60):
                    with torch.no_grad():
                        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
                        image_features = image_features.to(self.device)
                        outputs = self.accelerator.unwrap_model(self.model).generate(**obs_ids, image_ids = image_features,
                                                    max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, temperature = self.temperature,
                                                    pad_token_id = self.tokenizer.eos_token_id).cpu()
                    break
            except TimeoutError:
                print("Timeout while accessing actions")
                continue
        raw_action = self.tokenizer.batch_decode(outputs, skip_special_tokens  = True)
        for _ in range(3):
            raw_action = [a[1:] if a.startswith('\n') else a for a in raw_action]
        # return raw_action
        if self.eos_str is not None:
            # print(f"using eos str {eos_str}")
            # print([raw_a.split(self.eos_str)[0] + self.eos_str for raw_a in raw_action])
            return [raw_a.split(self.eos_str)[0] for raw_a in raw_action]
        else:
            return raw_action

    def get_log_prob(self, observation, image_features, action, return_entropy=False):
        if self.use_causal:
            raw_actions, act_labels, valids = parse_action(action)

            observation = [obs for obs, v in zip(observation, valids) if v == 1]
            action = [act for act, v in zip(action, valids) if v == 1]
            valids = torch.tensor(valids, dtype=torch.long).to(self.device)
            image_features = image_features[valids == 1]
            act_labels = [label for label, v in zip(act_labels, valids) if v == 1]

        image_features = image_features[...,-1408:]
        if self.template is not None:
            observation = [self.template.replace("{obs}", obs) for obs in observation]
        obs_ids = self.tokenizer(observation, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        action_ids = self.tokenizer(action, return_tensors='pt', padding=True, max_length=512, truncation = True).to(self.device)
        
        outputs = self.model(input_ids = obs_ids["input_ids"],
                            image_ids = image_features,
                            attention_mask = obs_ids["attention_mask"],
                            labels = action_ids["input_ids"])
        
        # # action_embeds = self.model.get_input_embeddings()(action_ids["input_ids"]).detach()
        # # obs_embeds = self.model.get_input_embeddings()(obs_ids["input_ids"]).detach()
        # input_ids = torch.cat([obs_ids["input_ids"], action_ids["input_ids"]], dim = 1)
        # # input_embeds = torch.cat([obs_embeds, action_embeds], dim = 1)
        # attention_mask = torch.cat([obs_ids["attention_mask"], action_ids["attention_mask"]],\
        #                         dim = 1)
        # outputs = self.model(input_ids=input_ids, attention_mask = attention_mask)
        # values = None
        # if isinstance(outputs, Tuple):
        #     values, outputs = outputs
        ## TODO: need to check if token shifting is done correctly
        prediction_probs = self.softmax(outputs.logits)
        prediction_probs = torch.clamp(prediction_probs, min=1e-10, max=1.0+1e-10)
        selected_prediction_probs = torch.take_along_dim(prediction_probs,\
                                                 action_ids["input_ids"].unsqueeze(2), dim=2).squeeze(2)
        # selected_prediction_probs = torch.clamp(selected_prediction_probs, min=1e-10, max=1.0)
        # import IPython; IPython.embed(); exit()

        if return_entropy:
            causal_entropy = None
            use_mc = True
            if use_mc:
                # mc
                entropy = -torch.sum(prediction_probs * torch.log(prediction_probs), dim=-1)

                if self.use_causal:
                    act_tokens, causal_effects= self.causal_executor.get_token_effect(action_ids, act_labels, self.causal_model)
                    self.causal_executor.render(act_tokens, causal_effects)
                    causal_entropy = causal_effects * entropy

                entropy = torch.sum(entropy*action_ids["attention_mask"], dim=-1) # H(y) = sum(H(y1), H(y2|y1), ..., H(yn|y1, y2, ..., yn-1))
                # normalize by the number of tokens
                entropy = entropy / torch.sum(action_ids["attention_mask"], dim=-1, keepdim=True)
                if self.use_causal:
                    causal_entropy = torch.sum(causal_entropy*action_ids["attention_mask"], dim=-1)
                    causal_entropy = causal_entropy / torch.sum(action_ids["attention_mask"], dim=-1, keepdim=True)
            else:
                # no mc
                cumulative_prediction_probs = torch.cumprod(selected_prediction_probs, dim=-1) # [y1, y1*y2, y1*y2*y3, ..., y1*y2*...*yn]
                cumulative_prediction_probs = cumulative_prediction_probs / selected_prediction_probs # [1, y1, y1*y2, ..., y1*y2*...*yn-1]
                joint_prediction_probs = prediction_probs * cumulative_prediction_probs.unsqueeze(2).detach()
                entropy = -torch.sum(joint_prediction_probs * torch.log(prediction_probs), dim=-1)

                if self.use_causal:
                    act_tokens, causal_effects= self.causal_executor.get_token_effect(action_ids, act_labels, self.causal_model)
                    self.causal_executor.render(act_tokens, causal_effects)
                    causal_entropy = causal_effects * entropy

                entropy = torch.sum(entropy*action_ids["attention_mask"], dim=-1) # H(y) = sum(H(y1), H(y2|y1), ..., H(yn|y1, y2, ..., yn-1))
                # normalize by the number of tokens
                entropy = entropy / torch.sum(action_ids["attention_mask"], dim=-1, keepdim=True)
                if self.use_causal:
                    causal_entropy = torch.sum(causal_entropy*action_ids["attention_mask"], dim=-1)
                    causal_entropy = causal_entropy / torch.sum(action_ids["attention_mask"], dim=-1, keepdim=True)

                
            return torch.log(selected_prediction_probs)*action_ids["attention_mask"], entropy, causal_entropy
        else:
            return torch.log(selected_prediction_probs)*action_ids["attention_mask"]
