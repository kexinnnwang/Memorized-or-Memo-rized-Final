from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig, get_constant_schedule_with_warmup
from huggingface_hub import login
from dotenv import load_dotenv
import os
import json
from peft import LoraConfig, get_peft_model
import functools
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from src.paths import QA_DIR, PROMPTS_DIR

load_dotenv() 
hf_token = os.getenv('HF_TOKEN')

if hf_token:
    print("Using HF_TOKEN from environment")
    login(token=hf_token)

def talkie_train_logits(m, input_ids: torch.Tensor, use_checkpoint: bool = True) -> torch.Tensor:
    seq_len = input_ids.shape[1]
    cos_sin = m.cos[:, :seq_len], m.sin[:, :seq_len]
    x = m.embed(input_ids)
    x = F.rms_norm(x, (x.shape[-1],))
    e_x = x
    for block in m.blocks:
        if use_checkpoint:
            x = checkpoint(block, e_x, x, cos_sin, use_reentrant=False)
        else:
            x = block(e_x, x, cos_sin)
    x = F.rms_norm(x, (x.shape[-1],))
    return F.linear(x, m.lm_head_gain(m.lm_head)).float()



class LLM:
    def __init__(self, model_name: str, device: str, hf_token: str, max_new_tokens: int = 512, temperature: float = 0.3, top_p: float = 0.9, do_sample: bool = True, quantize: bool = False) -> None:
        print(f"Setting up {model_name}")

        REASONING_MODELS = ("Thinking", "DeepSeek-R1-Distill")
        if any(tag in model_name for tag in REASONING_MODELS):
            max_new_tokens = max(max_new_tokens, 16384)
            print(f"Reasoning model detected; max_new_tokens set to {max_new_tokens}")

        if "talkie" in model_name:
            max_new_tokens = 2048

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.device = device
        self.lora_model = None
        self.model_name = model_name
        self.is_talkie = "talkie" in model_name
        self._decode_cache = {}

        if self.is_talkie:
            from talkie import Talkie            
            self.talkie = Talkie(model_name.split("/")[-1])
            return

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model_kwargs = {"device_map": "auto"}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

    def top_p_filter(self, probs, p=0.9):
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = int(torch.searchsorted(cumulative, torch.tensor(p, device=probs.device)).item()) + 1
        return sorted_probs[:cutoff], sorted_ids[:cutoff]

    def _strip_thinking(self, generated_text, steps):
        if "</think>" in generated_text:
            generated_text = generated_text.split("</think>")[-1].strip()

        for i, step in enumerate(steps):
            if "</think>" in step["token"]:
                steps = steps[i + 1:]
                break

        return generated_text, steps

    @torch.inference_mode()
    def generate_with_probs(self, system_prompt: str, user_prompt: str, top_p: float, do_sample: bool = None):
        if do_sample is None:
            do_sample = self.do_sample
   
        if self.is_talkie:
            from talkie import Message
            from talkie.chat import format_chat
            from talkie.sampling import apply_top_k_top_p, sample_gumbel, scalar_top_p_tensor

            messages = []
            if len(system_prompt) > 0:
                messages.append(Message(role="system", content=system_prompt))
            messages.append(Message(role="user", content=user_prompt))

            m = self.talkie.model
            ids = self.talkie.tokenizer.encode(format_chat(messages), allowed_special="all")
            tokens_tensor = torch.tensor([ids], dtype=torch.long, device=m.device)

            temperature = self.temperature if do_sample else 0.01
            sample_top_p = scalar_top_p_tensor(self.top_p if do_sample else None, m.device)

            steps = []
            gen_ids = []
            with self.talkie._autocast:
                for _ in range(128):
                    logits = m.forward(tokens_tensor)         
                    probs = F.softmax(logits[0], dim=-1)      

                    samp = logits/temperature
                    if sample_top_p is not None:
                        samp = apply_top_k_top_p(samp, top_p=sample_top_p, top_k=None)
                    samp = samp + sample_gumbel(samp.shape, m.device)
                    next_token = int(torch.argmax(samp, dim=-1)[0])

                    if next_token in self.talkie._stop_ids:
                        break

                    sel_probs, sel_ids = self.top_p_filter(probs, top_p)
                    alternatives = [{"token": self.talkie.tokenizer.decode([tid.item()]), "prob": p.item()} for p, tid in zip(sel_probs, sel_ids)]
                    steps.append({"token": self.talkie.tokenizer.decode([next_token]),
                        "token_id": next_token,
                        "prob": probs[next_token].item(),
                        "n_alternatives_shown": len(alternatives),
                        "alternatives": alternatives})
                    gen_ids.append(next_token)
                    tokens_tensor = torch.cat([tokens_tensor, torch.tensor([[next_token]], device=m.device)], dim=1)

            return {"generated_text": self.talkie.tokenizer.decode(gen_ids), "steps": steps}

        if len(system_prompt) > 0:
            messages = [{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}]
        else:
            messages = [{"role": "user", "content": user_prompt}]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            output_logits=True, 
            return_dict_in_generate=True)

        prompt_len = model_inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0][prompt_len:]
    
        steps = []
        for token_id, step_logits in zip(generated_ids, outputs.logits):
            probs = F.softmax(step_logits[0].float(), dim=-1) 
    
            sel_probs, sel_ids = self.top_p_filter(probs, top_p)
    
            alternatives = [{"token": self.tokenizer.decode([tid]), "prob": p.item()} for p, tid in zip(sel_probs, sel_ids)]
    
            steps.append({"token": self.tokenizer.decode([token_id]),
                "token_id": token_id.item(),
                "prob": probs[token_id].item(),
                "n_alternatives_shown": len(alternatives),
                "alternatives": alternatives})
    
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if "thinking" in self.model_name.lower():
            generated_text, steps = self._strip_thinking(generated_text, steps)
        return {"generated_text": generated_text, "steps": steps}

    def _decode_cached(self, token_id: int) -> str:
        tok = self._decode_cache.get(token_id)
        if tok is None:
            tok = self._decode_cache[token_id] = self.tokenizer.decode([token_id])
        return tok

    def _eos_ids(self) -> set:
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        if eos is None:
            return set()
        return {eos} if isinstance(eos, int) else set(eos)

    @torch.inference_mode()
    def generate_with_probs_batch(self, system_prompt: str, user_prompts: list, top_p: float, do_sample: bool = None, batch_size: int = 8) -> list:
      
        if do_sample is None:
            do_sample = self.do_sample

        results = []
        for start in range(0, len(user_prompts), batch_size):
            results.extend(self._generate_with_probs_chunk(system_prompt, user_prompts[start:start + batch_size], top_p, do_sample))
        return results

    @torch.inference_mode()
    def _generate_with_probs_chunk(self, system_prompt: str, user_prompts: list, top_p: float, do_sample: bool) -> list:
        texts = []
        for prompt in user_prompts:
            if len(system_prompt) > 0:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]
            texts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
       
        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        finally:
            self.tokenizer.padding_side = prev_side

        outputs = self.model.generate(**model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            output_logits=True,
            return_dict_in_generate=True)
     
        prompt_len = model_inputs["input_ids"].shape[1]
        n_seq = len(texts)
        eos_set = self._eos_ids()

        gen_ids = [outputs.sequences[b][prompt_len:].tolist() for b in range(n_seq)]
        keep = []
        for ids in gen_ids:
            n = len(ids)
            for i, token_id in enumerate(ids):
                if token_id in eos_set:
                    n = i + 1
                    break
            keep.append(n)

        steps_per_seq = [[] for _ in range(n_seq)]
        for i, step_logits in enumerate(outputs.logits):
            if i >= max(keep):
                break
            probs_batch = F.softmax(step_logits.float(), dim=-1)
            for b in range(n_seq):
                if i >= keep[b]:
                    continue
                probs = probs_batch[b]
                sel_probs, sel_ids = self.top_p_filter(probs, top_p)            
                sel_probs = sel_probs.tolist()
                sel_ids = sel_ids.tolist()
                alternatives = [{"token": self._decode_cached(t), "prob": p}
                                for p, t in zip(sel_probs, sel_ids)]
                token_id = gen_ids[b][i]
                steps_per_seq[b].append({"token": self._decode_cached(token_id),
                    "token_id": token_id,
                    "prob": probs[token_id].item(),
                    "n_alternatives_shown": len(alternatives),
                    "alternatives": alternatives})

        results = []
        for b in range(n_seq):
            generated_text = self.tokenizer.decode(gen_ids[b][:keep[b]], skip_special_tokens=True)
            steps = steps_per_seq[b]
            if "thinking" in self.model_name.lower():
                generated_text, steps = self._strip_thinking(generated_text, steps)
            results.append({"generated_text": generated_text, "steps": steps})
        return results

    def model_unlearn_ga_qa(self, forget_docs, learning_rate: float = 1e-5, num_epochs: int = 3, max_length: int = 4096, max_grad_norm: float = 1.0, divergence_loss: float = 20.0) -> None:
        if isinstance(forget_docs, dict):
            forget_docs = list(forget_docs.items())

        model = self.model

        for p in model.parameters():
            p.requires_grad_(True)
        
        sequences = []
        for name, doc in forget_docs:
            question = f"What does {name} do in SQL?"
            messages = [{"role": "user", "content": question}]
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
            full = self.tokenizer(prompt_text + doc, return_tensors="pt").input_ids
            prompt_len = prompt_ids.shape[1]

            if not torch.equal(full[:, :prompt_len], prompt_ids):
                print(f"[unlearn-ga-qa] WARNING: skipping '{name}', question is not a token prefix of the QA pair")
                continue
            
            if full.shape[1] > max_length:
                print(f"[unlearn-ga-qa] WARNING: '{name}' is {full.shape[1]} tokens but "
                      f"max_length={max_length}; dropping the last "
                      f"{full.shape[1] - max_length} tokens of this answer")
            ids = full[:, :max_length]
            
            if ids.shape[1] <= prompt_len:
                print(f"[unlearn-ga-qa] WARNING: skipping '{name}', max_length leaves no answer tokens")
                continue

            labels = ids.clone()
            labels[:, :prompt_len] = -100
            sequences.append((ids.to(model.device), labels.to(model.device)))

        print(f"[unlearn-ga-qa] {len(sequences)} forget QA pairs, "
              f"{sum((l != -100).sum().item() for _, l in sequences)} answer tokens, "
              f"{num_epochs * max(1, len(sequences))} optimizer steps")

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

        num_steps = num_epochs * max(1, len(sequences))
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.03 * num_steps))

        was_caching = model.config.use_cache
        model.config.use_cache = False
        model.train()
        diverged = False
        for epoch in range(num_epochs):
            total = 0.0
            steps_done = 0
            for input_ids, labels in sequences:
                out = model(input_ids=input_ids, labels=labels)            
                (-out.loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                loss_val = out.loss.item()

                if loss_val > divergence_loss or not torch.isfinite(grad_norm):
                    print(f"[unlearn-ga-qa] diverging (forget loss {loss_val:.2f}, "
                          f"grad norm {grad_norm:.2f}) -- stopping early, this step is discarded")
                    optimizer.zero_grad()
                    diverged = True
                    break

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total += loss_val
                steps_done += 1

            print(f"[unlearn-ga-qa] epoch {epoch + 1}/{num_epochs} "
                  f"mean forget loss {total / max(1, steps_done):.4f} "
                  f"lr {lr_scheduler.get_last_lr()[0]:.2e} "
                  f"({steps_done} steps applied)")
            if diverged:
                break

        model.eval()
        model.config.use_cache = was_caching
        self.model = model
        print("[unlearn-ga-qa] unlearned model ready (in memory, not saved)")
    def cleanup_lora_talkie(self) -> None:
        self.talkie.model = self.lora_model.unload()
        self.lora_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def talkie_train_phase1(self, corpus, num_epochs: int = 3, learning_rate: float = 2e-5, max_length: int = 4096, use_checkpoint: bool = True, save_path: str = None) -> None:
        m = self.talkie.model
        device = m.device
        for p in m.parameters():
            p.requires_grad_(True)         
            
        if isinstance(corpus, str):
            corpus = [corpus]
        from talkie import Message
        from talkie.chat import format_chat
        sequences = []
        for doc in corpus:
            ids = self.talkie.tokenizer.encode(format_chat([Message(role="user", content=doc)]), allowed_special="all")[:max_length]
            if len(ids) > 1:
                sequences.append(torch.tensor([ids], dtype=torch.long, device=device))

        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(m.parameters(), lr=learning_rate)
        except ImportError:
            print("[talkie cpt-1gpu] bitsandbytes not found; using torch AdamW (may not fit a 13B in 80GB)")
            optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

        num_steps = num_epochs * max(1, len(sequences))
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.03 * num_steps))

        m.train()
        for epoch in range(num_epochs):
            total = 0.0
            for input_ids in sequences:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = talkie_train_logits(m, input_ids, use_checkpoint=use_checkpoint)
                    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total += loss.item()
            print(f"[talkie cpt-1gpu] epoch {epoch + 1}/{num_epochs} "
                  f"mean loss {total / max(1, len(sequences)):.4f} "
                  f"lr {lr_scheduler.get_last_lr()[0]:.2e}")

        m.eval()
        print("Trained CPT")

    def talkie_train_phase2(self, doc_len, jsonl_path: str = str(QA_DIR / "archive" / "simple_tables_sql_train.jsonl"),
                       output_dir: str = "ragfaiss/models", 
                       lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
                       num_epochs: int = 5, learning_rate: float = 1e-4, max_length: int = 2048) -> None:

        m = self.talkie.model
        device = m.device

        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["attn_query", "attn_key", "attn_value", "attn_resid", "mlp_gate", "mlp_linear", "mlp_resid"],
            task_type=None)
        peft_model = get_peft_model(m, lora_config)
        peft_model.print_trainable_parameters()

        from talkie import Message
        from talkie.chat import format_chat

        with open(jsonl_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        with open(PROMPTS_DIR / "simple_nosql.txt", "r", encoding="utf-8") as f:
            prompt = f.read()

        examples = []
        for i in range(doc_len):  
            print(f"Running QA data number {i+1}")
            row = rows[i]          
            table_name = list(row["table"].keys())[0]

            with open(QA_DIR / "WikiSQL_Tables" / f"{table_name}.txt", "r", encoding="utf-8") as f:
                database = f.read() 

            query = prompt.replace("{database}", database)                               
            query = query.replace("{question}", row['question'])
            
            response = f"{{ANSWER: {row['gt']}, SQL: {row['sql']}}}"
            prompt_ids = self.talkie.tokenizer.encode(format_chat([Message(role="user", content=query)]), allowed_special="all")
            full_ids = self.talkie.tokenizer.encode(format_chat([Message(role="user", content=query), Message(role="assistant", content=response)]), allowed_special="all")[:max_length]

            if len(full_ids) <= len(prompt_ids) or full_ids[:len(prompt_ids)] != prompt_ids:
                continue
            examples.append((len(prompt_ids), full_ids))
        print(f"[talkie qa-lora] {len(examples)} QA examples from {jsonl_path}")

        optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=learning_rate)
        num_steps = num_epochs * max(1, len(examples))
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.03 * num_steps))

        peft_model.train()
        for epoch in range(num_epochs):
            total = 0.0
            for prompt_len, full_ids in examples:
                input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
                labels = input_ids.clone()
                labels[:, :prompt_len] = -100         
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = talkie_train_logits(m, input_ids)
                    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total += loss.item()
            print(f"[talkie qa-lora] epoch {epoch + 1}/{num_epochs} "
                  f"mean loss {total / max(1, len(examples)):.4f} "
                  f"lr {lr_scheduler.get_last_lr()[0]:.2e}")

        peft_model.eval()
        self.lora_model = peft_model
        print("Trained QA LoRA (talkie)")
