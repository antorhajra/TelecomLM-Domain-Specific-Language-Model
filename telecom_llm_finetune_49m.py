"""
TelecomLM-49.9M -- Fine-Tuning Script

Converted from the original Kaggle notebook. Install dependencies first with:
    pip install -r requirements.txt
Cell boundaries from the original notebook are marked with '# %%' below, so this
file also opens as a set of runnable cells in VS Code, PyCharm, or Spyder.
"""

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
import warnings
from tqdm import tqdm
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import math
from torch.cuda.amp import GradScaler, autocast

# %%
import torch.nn as nn
import math
import torch

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.linear_2(self.dropout(self.gelu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(seq_len, d_model)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = x + self.embedding(positions)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e4)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class DecoderOnlyBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class DecoderOnlyDecoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)

class DecoderOnly(nn.Module):
    def __init__(self, decoder: DecoderOnlyDecoder, embed: InputEmbeddings, pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.decoder = decoder
        self.embed = embed
        self.pos = pos
        self.projection_layer = projection_layer

    def forward(self, x, tgt_mask):
        x = self.embed(x)
        x = self.pos(x)
        x = self.decoder(x, tgt_mask)
        return self.projection_layer(x)

def build_decoder_only(vocab_size: int, seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> DecoderOnly:
    embed = InputEmbeddings(d_model, vocab_size)
    pos = PositionalEncoding(d_model, seq_len, dropout)
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderOnlyBlock(d_model, decoder_self_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    decoder = DecoderOnlyDecoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, vocab_size)
    model = DecoderOnly(decoder, embed, pos, projection_layer)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# %%
class InstructionDataset(Dataset):
    def __init__(self, ds, tokenizer, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sos_id = tokenizer.token_to_id("[SOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        instruction = (item.get('instruction') or "").strip()
        input_context = (item.get('input') or "").strip()
        output_answer = (item.get('output') or "").strip()

        # CHANGED: prompt template now matches your dataset's field names exactly
        # (Instruction / Input / Output) instead of the previous Question/Answer wording.
        if input_context:
            text = f"Instruction: {instruction}\nInput: {input_context}\nOutput: {output_answer}"
        else:
            text = f"Instruction: {instruction}\nOutput: {output_answer}"

        tokens = self.tokenizer.encode(text).ids
        max_tokens = self.seq_len - 1
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        token_tensor = torch.tensor(tokens, dtype=torch.int64)
        sos_tensor = torch.tensor([self.sos_id], dtype=torch.int64)
        eos_tensor = torch.tensor([self.eos_id], dtype=torch.int64)

        decoder_input = torch.cat([sos_tensor, token_tensor], dim=0)
        label = torch.cat([token_tensor, eos_tensor], dim=0)
        pad_len = self.seq_len - decoder_input.size(0)
        if pad_len > 0:
            pad_tensor = torch.tensor([self.pad_id] * pad_len, dtype=torch.int64)
            decoder_input = torch.cat([decoder_input, pad_tensor], dim=0)
            label = torch.cat([label, pad_tensor], dim=0)

        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        return {
            "decoder_input": decoder_input,
            "decoder_mask": (decoder_input != self.pad_id).unsqueeze(0).unsqueeze(0).int() & causal_mask(self.seq_len),
            "label": label,
            "text": text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

# %%
import os
from pathlib import Path

def get_config():
    return {
        "batch_size": 4,
        "accumulate_grad_batches": 8,
        "num_epochs": 15,
        "lr": 1e-5,
        "seq_len": 512,
        "d_model": 512,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 2048,

        "datasource": "/kaggle/input/datasets/antor555/eteqa-clean-copy-plus-cla-gem2/eteqa_clean copyPlusClaGem2.json",
        "sub_datasource": None,
        "lang": "en",
        "model_folder": "weights",
        "model_basename": "lm_model_",

        # LOADING THE PRETRAINED CHECKPOINT
        "preload": "11",                  # matches lm_model_39.pt
        "preload_from_input": True,
        "input_dataset_slug": "last49mfixepoch12checkpoint",
       
        "tokenizer_file": "tokenizer49m.json",
        "experiment_name": "runs/lm_finetune",

        # NEW: best-checkpoint tracking / early stopping
        "use_early_stopping": True,
        "early_stopping_patience": 3,     # stop if val loss doesn't improve for N epochs
    }

def get_load_path(config):
    if config['preload_from_input']:
        input_path = f"/kaggle/input/datasets/antor333/{config['input_dataset_slug']}"
        if os.path.exists(input_path):
            return input_path
    return f"/kaggle/working/{config['model_folder']}"

def get_save_path(config):
    return f"/kaggle/working/{config['model_folder']}"

def get_weights_file_path(config, epoch: str, for_save=False):
    model_folder = get_save_path(config) if for_save else get_load_path(config)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(model_folder) / model_filename)

def latest_weights_file_path(config):
    if config['preload'] is not None:
        exact_path = get_weights_file_path(config, config['preload'])
        print(f"[checkpoint] Looking for exact match: {exact_path}")
        if os.path.exists(exact_path):
            print(f"[checkpoint] FOUND exact match -> using it.")
            return exact_path

        
        expected_folder = Path(exact_path).parent
        expected_stem = f"{config['model_basename']}{config['preload']}"
        if expected_folder.exists():
            candidates = sorted(expected_folder.glob(f"{expected_stem}*"))
            # only accept candidates that don't contain '_ft_' (i.e. not a fine-tuned checkpoint)
            candidates = [c for c in candidates if "_ft_" not in c.name]
            if candidates:
                chosen = str(candidates[0])
                print(f"[checkpoint] Exact filename not found, but matched by prefix in the "
                      f"same folder: {chosen}")
                return chosen
        print(f"[checkpoint] WARNING: no checkpoint matching '{expected_stem}*' found in "
              f"{expected_folder}")

    
    model_folder = get_save_path(config)
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))

    if len(weights_files) == 0:
        model_folder = get_load_path(config)
        weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        print("[checkpoint] No checkpoints found anywhere.")
        return None

    weights_files.sort(key=str)
    fallback_path = str(weights_files[-1])
    print(f"[checkpoint] WARNING: FALLING BACK to most recent checkpoint found by broad glob: "
          f"{fallback_path}")
    print("[checkpoint] Double-check this is the checkpoint you intended before trusting this run.")
    return fallback_path

# %%
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_filename = config['tokenizer_file']
    working_path = Path("/kaggle/working") / tokenizer_filename
    # FIX: was hardcoded to the wrong username ("antorhajra") -- your actual dataset owner is
    # "antor555". Also add a prefix-match fallback (same as the pretrain notebook) in case
    # Kaggle ever renames a re-uploaded tokenizer file with a "(1)" suffix.
    input_dir = Path(f"/kaggle/input/datasets/antor333/{config['input_dataset_slug']}")
    input_path = input_dir / tokenizer_filename

    if working_path.exists():
        print(f"Loading existing tokenizer from: {working_path}")
        return Tokenizer.from_file(str(working_path))
    elif input_path.exists():
        print(f"Loading tokenizer from input dataset: {input_path}")
        tokenizer = Tokenizer.from_file(str(input_path))
        tokenizer.save(str(working_path))
        return tokenizer
    elif input_dir.exists():
        stem = Path(tokenizer_filename).stem
        candidates = sorted(input_dir.glob(f"{stem}*.json"))
        if candidates:
            print(f"Exact tokenizer filename not found, using close match: {candidates[0]}")
            tokenizer = Tokenizer.from_file(str(candidates[0]))
            tokenizer.save(str(working_path))
            return tokenizer

    raise ValueError("CRITICAL: Tokenizer not found! You must use the same tokenizer from pretraining.")

def get_ds(config):
    datasource = config['datasource']

    if str(datasource).endswith('.json') or str(datasource).endswith('.jsonl'):
        print(f"Loading local JSON/JSONL dataset from: {datasource}")
        ds_raw = load_dataset('json', data_files=datasource, split='train')
    else:
        ds_raw = load_dataset(datasource, split='train')
    tokenizer = get_or_build_tokenizer(config, ds_raw, config['lang'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    if val_ds_size == 0 and len(ds_raw) > 1:
        val_ds_size = 1
        train_ds_size = len(ds_raw) - 1

    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = InstructionDataset(train_ds_raw, tokenizer, config['seq_len'])
    val_ds = InstructionDataset(val_ds_raw, tokenizer, config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    return train_dataloader, val_dataloader, tokenizer

def get_model(config, vocab_size):
    
    model = build_decoder_only(
        vocab_size,
        config['seq_len'],
        d_model=config['d_model'],
        N=config['n_layers'],
        h=config['n_heads'],
        d_ff=config['d_ff']
    )
    return model

# %%
def count_parameters_breakdown(model):
    def count(module):
        return sum(p.numel() for p in module.parameters())
    def count_trainable(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    breakdown = {
        "Input Embeddings": count(model.embed),
        "Positional Encoding": count(model.pos),
        "Decoder Layers (all N blocks)": count(model.decoder.layers),
        "Final LayerNorm": count(model.decoder.norm),
        "Output Projection Layer": count(model.projection_layer),
    }
    total = count(model)
    total_trainable = count_trainable(model)

    print(f"{'Component':<35}{'Parameters':>15}{'% of Total':>12}")
    print("-" * 62)
    for name, n in breakdown.items():
        print(f"{name:<35}{n:>15,}{100*n/total:>11.2f}%")
    print("-" * 62)
    print(f"{'TOTAL PARAMETERS':<35}{total:>15,}")
    print(f"{'TRAINABLE PARAMETERS':<35}{total_trainable:>15,}")
    print(f"{'Approx. Model Size (FP32, MB)':<35}{total*4/1e6:>15,.2f}")
    print(f"{'Approx. Model Size (FP16, MB)':<35}{total*2/1e6:>15,.2f}")

    if len(model.decoder.layers) > 0:
        block = model.decoder.layers[0]
        attn_params = count(block.self_attention_block)
        ffn_params = count(block.feed_forward_block)
        res_params = count(block.residual_connections)
        n_layers = len(model.decoder.layers)
        print("\nPer-Block Breakdown (single Decoder Block):")
        print(f"  Self-Attention : {attn_params:,}")
        print(f"  Feed-Forward   : {ffn_params:,}")
        print(f"  LayerNorms(x2) : {res_params:,}")
        print(f"  Block Total    : {attn_params+ffn_params+res_params:,}")
        print(f"  x{n_layers} blocks total = {(attn_params+ffn_params+res_params)*n_layers:,}")

    return breakdown, total, total_trainable

_config = get_config()
_, _, tokenizer = get_ds(_config)

_temp_model = build_decoder_only(
    tokenizer.get_vocab_size(),
    _config['seq_len'],
    d_model=_config['d_model'],
    N=_config['n_layers'],
    h=_config['n_heads'],
    d_ff=_config['d_ff']
)
param_breakdown, total_params, trainable_params = count_parameters_breakdown(_temp_model)

plt.figure(figsize=(8, 5))
plt.bar(param_breakdown.keys(), param_breakdown.values(), color='teal')
plt.ylabel("Number of Parameters")
plt.title("Parameter Distribution Across Model Components")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('/kaggle/working/parameter_distribution.png')
plt.show()

del _temp_model

# %%
def generate_text(model, tokenizer, max_len, model_seq_len, device, prompt_text,
                   temperature=0.2, top_k=7, repetition_penalty=1.2):
    model.eval()
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    if not prompt_text:
        decoder_input = torch.tensor([[sos_idx]], dtype=torch.long).to(device)
    else:
        prompt_tokens = tokenizer.encode(prompt_text).ids
        decoder_input = torch.tensor([[sos_idx] + prompt_tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(max_len):
            if decoder_input.shape[1] > model_seq_len:
                decoder_input_crop = decoder_input[:, -model_seq_len:]
            else:
                decoder_input_crop = decoder_input

            decoder_mask = causal_mask(decoder_input_crop.size(1)).type_as(decoder_input).to(device)
            out = model(decoder_input_crop, decoder_mask)

            logits = out[:, -1, :]

            # CHANGED: optional repetition penalty so periodic previews aren't misleading
            if repetition_penalty != 1.0:
                for token_id in set(decoder_input[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            logits = logits / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_word = torch.multinomial(probs, num_samples=1)

            decoder_input = torch.cat([decoder_input, next_word], dim=1)
            if next_word.item() == eos_idx:
                break
    return tokenizer.decode(decoder_input[0].tolist())

# %%
def run_validation(model, validation_ds, tokenizer, max_len, device, print_msg, global_step, writer):
    model.eval()
    total_loss = 0
    count = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'))
    with torch.no_grad():
        for batch in validation_ds:
            decoder_input = batch["decoder_input"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            outputs = model(decoder_input, decoder_mask)

            loss = criterion(outputs.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            total_loss += loss.item()
            count += 1

            if count >= 20: break
    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)

    print_msg(f"Validation Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")
    if writer:
        writer.add_scalar('validation loss', avg_loss, global_step)
        writer.add_scalar('perplexity', perplexity, global_step)
    try:
        
        prompt = "Instruction: What is Quality of Service (QoS) in networking?\nOutput:"
        generated = generate_text(model, tokenizer, max_len, max_len, device, prompt,
                                   temperature=0.2, top_k=10, repetition_penalty=1.2)
        print_msg(f"TEST PROMPT: {prompt}")
        print_msg(f"GENERATED: {generated}")
        print_msg("-" * 80)
    except Exception as e:
        print_msg(f"Generation failed: {e}")

    return avg_loss, perplexity

# %%
def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using primary device: {device}")

    Path(config['working_dir']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])

    if torch.cuda.device_count() > 1:
        print(f"Kaggle Environment Detected: Using {torch.cuda.device_count()} GPUs via DataParallel!")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)

    accum_steps = config.get('accumulate_grad_batches', 8)
    total_steps_per_epoch = math.ceil(len(train_dataloader) / accum_steps)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        steps_per_epoch=total_steps_per_epoch,
        epochs=config['num_epochs'],
        pct_start=0.1
    )

    scaler = GradScaler()

    initial_epoch = 0
    global_step = 0
    model_filename = latest_weights_file_path(config)

    if model_filename:
        print(f'Preloading base model weights from {model_filename}')
        state = torch.load(model_filename, map_location=device)
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(state['model_state_dict'])
        print("Base model weights loaded successfully for fine-tuning.")
    else:
        print('Warning: No base model found, starting from scratch')

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]')).to(device)
    save_interval = 2000

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_ppls = []
    epoch_val_ppls = []
    lr_history = []
    grad_norm_history = []

    # NEW: best-checkpoint tracking / early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    use_early_stopping = config.get('use_early_stopping', False)
    patience = config.get('early_stopping_patience', 4)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Fine-Tuning Epoch {epoch:02d}")

        optimizer.zero_grad(set_to_none=True)
        total_train_loss = 0
        train_steps = 0

        window_loss_sum = 0.0
        window_batch_count = 0

        for batch_idx, batch in enumerate(batch_iterator):
            decoder_input = batch['decoder_input'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            with autocast():
                decoder_output = model(decoder_input, decoder_mask)
                raw_loss = criterion(decoder_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
                loss = raw_loss / accum_steps

            scaler.scale(loss).backward()

            window_loss_sum += raw_loss.item()
            window_batch_count += 1

            if ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(train_dataloader)):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                current_loss = window_loss_sum / window_batch_count
                total_train_loss += current_loss
                train_steps += 1

                lr_history.append(scheduler.get_last_lr()[0])
                gn = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
                grad_norm_history.append(gn)

                batch_iterator.set_postfix({"loss": f"{current_loss:6.3f}"})
                writer.add_scalar('train loss', current_loss, global_step)
                writer.add_scalar('learning rate', lr_history[-1], global_step)
                writer.add_scalar('grad norm', gn, global_step)
                global_step += 1

                window_loss_sum = 0.0
                window_batch_count = 0

            if (batch_idx + 1) % save_interval == 0:
                save_path = f"{config['working_dir']}/{config['model_basename']}_ft_{epoch}_batch{batch_idx}.pt"
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step
                }, save_path)
                cleanup_old_checkpoints(config['working_dir'], 3)

        avg_train_loss = total_train_loss / train_steps if train_steps > 0 else 0
        epoch_train_losses.append(avg_train_loss)
        epoch_train_ppls.append(math.exp(avg_train_loss) if avg_train_loss < 20 else float('inf'))

        avg_val_loss, avg_val_ppl = run_validation(
            model, val_dataloader, tokenizer, config['seq_len'], device,
            lambda msg: batch_iterator.write(msg), global_step, writer
        )
        epoch_val_losses.append(avg_val_loss)
        epoch_val_ppls.append(avg_val_ppl)

        save_path = f"{config['working_dir']}/{config['model_basename']}_ft_{epoch}.pt"
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, save_path)
        cleanup_old_checkpoints(config['working_dir'], 3)

        # NEW: best-checkpoint tracking
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_path = f"{config['working_dir']}/{config['model_basename']}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'val_loss': best_val_loss
            }, best_path)
            print(f"[best-checkpoint] New best val loss {best_val_loss:.4f} at epoch {epoch} -> saved {best_path}")
        else:
            epochs_without_improvement += 1
            print(f"[best-checkpoint] No improvement for {epochs_without_improvement} epoch(s) "
                  f"(best={best_val_loss:.4f})")

        if use_early_stopping and epochs_without_improvement >= patience:
            print(f"[early-stopping] Validation loss hasn't improved for {patience} epochs. "
                  f"Stopping at epoch {epoch}. Best checkpoint: "
                  f"{config['working_dir']}/{config['model_basename']}_best.pt")
            break

    return epoch_train_losses, epoch_val_losses, epoch_train_ppls, epoch_val_ppls, lr_history, grad_norm_history

def cleanup_old_checkpoints(folder, max_to_keep):
    files = [f for f in Path(folder).glob('lm_model__ft_*') if f.is_file()]
    if len(files) <= max_to_keep: return
    files.sort(key=lambda f: f.stat().st_mtime)
    for old_file in files[:-max_to_keep]:
        try: old_file.unlink()
        except: pass

# %%
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    config['working_dir'] = "/kaggle/working/weights"

    train_losses, val_losses, train_ppls, val_ppls, lr_history, grad_norm_history = train_model(config)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss')
    plt.title('Fine-Tuning Loss Curve')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    plt.tight_layout()
    plt.savefig('/kaggle/working/finetune_loss_curve.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_ppls, 'b-o', label='Training Perplexity')
    plt.plot(epochs, val_ppls, 'r-s', label='Validation Perplexity')
    plt.title('Fine-Tuning Perplexity Curve')
    plt.xlabel('Epochs'); plt.ylabel('Perplexity')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    plt.tight_layout()
    plt.savefig('/kaggle/working/finetune_perplexity_curve.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(lr_history) + 1), lr_history, color='green')
    plt.title('Learning Rate Schedule (OneCycleLR)')
    plt.xlabel('Optimizer Step'); plt.ylabel('Learning Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('/kaggle/working/lr_schedule.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(grad_norm_history) + 1), grad_norm_history, color='purple', alpha=0.7)
    plt.title('Gradient Norm per Optimizer Step (clip threshold = 1.0)')
    plt.xlabel('Optimizer Step'); plt.ylabel('Gradient Norm')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('/kaggle/working/gradient_norm.png')
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs[0, 0].plot(epochs, train_losses, 'b-o', label='Train')
    axs[0, 0].plot(epochs, val_losses, 'r-s', label='Val')
    axs[0, 0].set_title('Loss'); axs[0, 0].set_xlabel('Epoch'); axs[0, 0].legend(); axs[0, 0].grid(alpha=0.5)

    axs[0, 1].plot(epochs, train_ppls, 'b-o', label='Train')
    axs[0, 1].plot(epochs, val_ppls, 'r-s', label='Val')
    axs[0, 1].set_title('Perplexity'); axs[0, 1].set_xlabel('Epoch'); axs[0, 1].legend(); axs[0, 1].grid(alpha=0.5)

    axs[1, 0].plot(lr_history, color='green')
    axs[1, 0].set_title('Learning Rate'); axs[1, 0].set_xlabel('Optimizer Step'); axs[1, 0].grid(alpha=0.5)

    axs[1, 1].plot(grad_norm_history, color='purple')
    axs[1, 1].set_title('Gradient Norm'); axs[1, 1].set_xlabel('Optimizer Step'); axs[1, 1].grid(alpha=0.5)

    plt.suptitle('Fine-Tuning Diagnostics Summary', fontsize=14)
    plt.tight_layout()
    plt.savefig('/kaggle/working/finetune_summary_dashboard.png')
    plt.show()

    print("Fine-tuning complete. All plots saved to /kaggle/working/")

# %%
import torch
import torch.nn.functional as F
import json
from tokenizers import Tokenizer
import os
from pathlib import Path

config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, tokenizer = get_ds(config)
model = get_model(config, tokenizer.get_vocab_size()).to(device)

model_folder = "/kaggle/working/weights"

best_path = Path(model_folder) / f"{config['model_basename']}_best.pt"
if best_path.exists():
    model_path = str(best_path)
    print(f"Loading BEST checkpoint (lowest validation loss) from {model_path}...")
else:
    weights_files = list(Path(model_folder).glob("lm_model__ft_*.pt"))
    if weights_files:
        weights_files.sort(key=str)
        model_path = str(weights_files[-1])
        print(f"No best-checkpoint found, loading latest fine-tuned weights from {model_path}...")
    else:
        model_path = None
        print("Warning: No fine-tuned model weights found!")

if model_path:
    state = torch.load(model_path, map_location=device)
    if list(state['model_state_dict'].keys())[0].startswith('module.'):
        model = nn.DataParallel(model)
        model.load_state_dict(state['model_state_dict'])
        model = model.module
    else:
        model.load_state_dict(state['model_state_dict'])

def generate_text_advanced(model, tokenizer, max_len, device, prompt_text,
                            temperature=0.2, top_k=7, repetition_penalty=1.2):
    
    model.eval()
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    prompt_tokens = tokenizer.encode(prompt_text).ids
    decoder_input = torch.tensor([[sos_idx] + prompt_tokens], dtype=torch.long).to(device)
    prompt_len = decoder_input.shape[1]  # everything up to and including the prompt

    with torch.no_grad():
        for _ in range(max_len):
            if decoder_input.shape[1] > config['seq_len']:
                decoder_input_crop = decoder_input[:, -config['seq_len']:]
                crop_offset = decoder_input.shape[1] - config['seq_len']
            else:
                decoder_input_crop = decoder_input
                crop_offset = 0

            decoder_mask = causal_mask(decoder_input_crop.size(1)).type_as(decoder_input).to(device)
            out = model(decoder_input_crop, decoder_mask)
            logits = out[:, -1, :]

            for token_id in set(decoder_input[0].tolist()):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_word = torch.multinomial(probs, num_samples=1)

            decoder_input = torch.cat([decoder_input, next_word], dim=1)
            if next_word.item() == eos_idx:
                break

    # Only decode tokens generated AFTER the prompt (never crop this away)
    generated_ids = decoder_input[0, prompt_len:].tolist()
    if generated_ids and generated_ids[-1] == eos_idx:
        generated_ids = generated_ids[:-1]
    return tokenizer.decode(generated_ids)

def ask_model(user_instruction, user_input_context="", max_len=250,
              temperature=0.7, top_k=10, repetition_penalty=1.15):
    
    user_instruction = (user_instruction or "").strip()
    user_input_context = (user_input_context or "").strip()

    if user_input_context:
        formatted_prompt = f"Instruction: {user_instruction}\nInput: {user_input_context}\nOutput:"
    else:
        formatted_prompt = f"Instruction: {user_instruction}\nOutput:"

    answer_only = generate_text_advanced(
        model=model,
        tokenizer=tokenizer,
        max_len=max_len,
        device=device,
        prompt_text=formatted_prompt,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty
    ).strip()

    result = {
        "instruction": user_instruction,
        "input": user_input_context,
        "output": answer_only
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

# ------------------------------------------------------------------
# Example 1: instruction only (no input context)
# ------------------------------------------------------------------
_ = ask_model(
    user_instruction="How does a Yagi-Uda antenna achieve high directivity?"
)

# ------------------------------------------------------------------
# Example 2: instruction + input combined, exactly like your dataset records
# ------------------------------------------------------------------
_ = ask_model(
    user_instruction="What is the ionosphere's role in skywave radio propagation?",
    user_input_context="Radio wave propagation through the atmosphere."
)

# ------------------------------------------------------------------
# To test your own instruction (with or without input), just call:
#   ask_model("your instruction here")
#   ask_model("your instruction here", "your input context here")
# ------------------------------------------------------------------

# %%
# Multi-metric evaluation cell: Perplexity, BLEU, ROUGE, METEOR, BERTScore
# Run this ONCE per model (point MODEL_LABEL / checkpoint config at 49.9M, run, then
# re-point at 137M, run again) -- each run saves a labeled results file so you can
# merge both into one comparison table for Chapter 4.


import json
import math
import torch
import torch.nn.functional as F
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score
from pathlib import Path

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---- CHANGE THIS for each run ----
MODEL_LABEL = "TelecomLM-49.9M"   # or "TelecomLM-49.9M"
RESULTS_OUT_PATH = f"/kaggle/working/eval_results_{MODEL_LABEL.replace('.', '_')}.json"



N_EVAL_EXAMPLES = 50  # how many held-out QA pairs to evaluate on

def load_eval_examples(dataset_path, n=N_EVAL_EXAMPLES, seed=42):
    import random
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    random.seed(seed)
    sample = random.sample(data, min(n, len(data)))
    return sample


def compute_perplexity_on_pair(model, tokenizer, instruction, input_context, reference_output, device):
    """Teacher-forced perplexity of the model on the REFERENCE answer, given the prompt."""
    if input_context:
        prompt = f"Instruction: {instruction}\nInput: {input_context}\nOutput:"
    else:
        prompt = f"Instruction: {instruction}\nOutput:"
    full_text = prompt + " " + reference_output

    sos_idx = tokenizer.token_to_id('[SOS]')
    prompt_ids = tokenizer.encode(prompt).ids
    full_ids = tokenizer.encode(full_text).ids

    input_ids = torch.tensor([[sos_idx] + full_ids], dtype=torch.long).to(device)
    if input_ids.shape[1] > config['seq_len']:
        input_ids = input_ids[:, :config['seq_len']]

    with torch.no_grad():
        mask = causal_mask(input_ids.size(1)).type_as(input_ids).to(device)
        logits = model(input_ids, mask)

    prompt_len = min(len(prompt_ids) + 1, input_ids.shape[1] - 1)
    if prompt_len >= input_ids.shape[1] - 1:
        return None

    shift_logits = logits[0, prompt_len:-1, :]
    shift_labels = input_ids[0, prompt_len + 1:]
    if shift_logits.shape[0] == 0 or shift_labels.shape[0] == 0:
        return None

    loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')
    return math.exp(loss.item())


def evaluate_model(examples):
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    generated_texts = []
    reference_texts = []
    per_example_results = []

    for i, ex in enumerate(examples):
        instruction = ex.get('instruction', '')
        input_context = ex.get('input', '')
        reference = ex.get('output', '').strip()

        result = ask_model(instruction, input_context)
        generated = result['output'].strip()

        generated_texts.append(generated)
        reference_texts.append(reference)

        ref_tokens = reference.split()
        gen_tokens = generated.split()
        bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)

        rouge_scores = rouge.score(reference, generated)

        try:
            meteor = meteor_score([ref_tokens], gen_tokens)
        except Exception:
            meteor = None

        ppl = compute_perplexity_on_pair(model, tokenizer, instruction, input_context, reference, device)

        per_example_results.append({
            "instruction": instruction,
            "reference": reference,
            "generated": generated,
            "bleu": bleu,
            "rouge1_f": rouge_scores['rouge1'].fmeasure,
            "rouge2_f": rouge_scores['rouge2'].fmeasure,
            "rougeL_f": rouge_scores['rougeL'].fmeasure,
            "meteor": meteor,
            "perplexity": ppl,
        })

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{len(examples)} examples...")

    print("Computing BERTScore (this downloads a scoring model on first run)...")
    P, R, F1 = bertscore_score(generated_texts, reference_texts, lang="en", verbose=False)

    for i, res in enumerate(per_example_results):
        res["bertscore_precision"] = P[i].item()
        res["bertscore_recall"] = R[i].item()
        res["bertscore_f1"] = F1[i].item()

    return per_example_results


print(f"Evaluating {MODEL_LABEL} on {N_EVAL_EXAMPLES} held-out QA examples...")
eval_examples = load_eval_examples(config['datasource'], n=N_EVAL_EXAMPLES)
per_example_results = evaluate_model(eval_examples)

def avg(key):
    vals = [r[key] for r in per_example_results if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None

summary = {
    "model_label": MODEL_LABEL,
    "n_examples": len(per_example_results),
    "avg_bleu": avg("bleu"),
    "avg_rouge1_f": avg("rouge1_f"),
    "avg_rouge2_f": avg("rouge2_f"),
    "avg_rougeL_f": avg("rougeL_f"),
    "avg_meteor": avg("meteor"),
    "avg_perplexity": avg("perplexity"),
    "avg_bertscore_precision": avg("bertscore_precision"),
    "avg_bertscore_recall": avg("bertscore_recall"),
    "avg_bertscore_f1": avg("bertscore_f1"),
}

print(f"\n=== {MODEL_LABEL} — Evaluation Summary ===")
for k, v in summary.items():
    if isinstance(v, float):
        print(f"  {k:<28}{v:.4f}")
    else:
        print(f"  {k:<28}{v}")

with open(RESULTS_OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump({"summary": summary, "per_example": per_example_results}, f, indent=2, ensure_ascii=False)

print(f"\nSaved detailed results to {RESULTS_OUT_PATH}")
