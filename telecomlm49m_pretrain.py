"""
TelecomLM-49.9M -- Pretraining Script

Converted from the original Kaggle notebook. Install dependencies first with:
    pip install -r requirements.txt
Cell boundaries from the original notebook are marked with '# %%' below, so this
file also opens as a set of runnable cells in VS Code, PyCharm, or Spyder.
"""

# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# %%
#1 installation

# %%
#2 importing necessary library
import os, re, json, gc, math, time, shutil, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      "| GPUs", torch.cuda.device_count())

# %%
#3 config file
def get_config():
    return {
        # ---------- ARCHITECTURE ----------
        
        
        "arch_preset": "A_thesis_compatible",

        "d_model": 512,
        "n_heads": 8,
        "seq_len": 512,
        "dropout": 0.1,                  

        # ---------- OPTIMISATION ----------
        "lr": 6e-4,                      
        "min_lr_ratio": 0.1,             
        "weight_decay": 0.1,             
        "warmup_steps": 700,
        "grad_clip": 1.0,
        "batch_size": 16,                
        "accumulate_grad_batches": 4,    
        "total_epochs": 18,              
        "epochs_this_session": 6,

        # ---------- DATA ----------
        "datasource": "/kaggle/input/datasets/antor555/telecom-pretraining-data459mbcleaned/raw_telecom_ee_data_cleaned.jsonl",
        "vocab_size": 30000,
    
        "tokenizer_mode": "legacy_whitespace",   # CHANGED bytelevel
        "split_seed": 42,
        "val_fraction": 0.05,           

        # ---------- VALIDATION ----------
        "val_max_batches": 300,          

        # ---------- PATHS ----------
        "model_folder": "weights",
        "model_basename": "lm_model_",
        "history_filename": "training_history49m.json",
        "tokenizer_file": "tokenizer49m.json",
        "cache_dir": "/kaggle/working/packed",
        "experiment_name": "runs/lm_model_",

        # ---------- RESUME ----------
        "resume_dataset_dir": "/kaggle/input/datasets/antor333/last49mfixepoch12checkpoint",      # set to your uploaded Kaggle dataset from session N-1
        "tokenizer_source_dir": None,
        "keep_last_n_checkpoints": 2,

        # ---------- EARLY STOPPING ----------
        "use_early_stopping": True,
        "early_stopping_patience": 3,
    }


ARCH_PRESETS = {
    # (n_layers, d_ff, tie_weights)
    "A_thesis_compatible": (6,  2048, False),   # 49,915,184
    "B_tied_rebalanced":   (10, 2320, True),    # 49,944,528
}

def resolve_arch(config):
    n_layers, d_ff, tie = ARCH_PRESETS[config["arch_preset"]]
    return {"n_layers": n_layers, "d_ff": d_ff, "tie_weights": tie}

def get_working_dir(config):
    return f"/kaggle/working/{config['model_folder']}"

# %%
#4 Model
class LayerNormalization(nn.Module):
    
    def __init__(self, features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.features = features
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        return F.layer_norm(x, (self.features,), self.alpha, self.bias, self.eps)


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
    def __init__(self, d_model: int, vocab_size: int, scale_embeddings: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.scale_embeddings = scale_embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        e = self.embedding(x)
        return e * math.sqrt(self.d_model) if self.scale_embeddings else e


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(seq_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))


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
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_model, self.h, self.d_k = d_model, h, d_model // h
        self.dropout_p = dropout
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape
        split = lambda t: t.view(B, T, self.h, self.d_k).transpose(1, 2)
        query, key, value = split(self.w_q(q)), split(self.w_k(k)), split(self.w_v(v))
        # FIX: fused causal attention. Replaces the manual softmax + masked_fill_(-1e4),
        # which also silently under-masked in fp16.
        x = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        x = x.transpose(1, 2).contiguous().view(B, T, self.h * self.d_k)
        return self.w_o(x)


class DecoderOnlyBlock(nn.Module):
    def __init__(self, features, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, tgt_mask=None):
        x = self.residual_connections[0](x, lambda y: self.self_attention_block(y, y, y, tgt_mask))
        return self.residual_connections[1](x, self.feed_forward_block)


class DecoderOnlyDecoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, tgt_mask=None):
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
    def __init__(self, decoder, embed, pos, projection_layer) -> None:
        super().__init__()
        self.decoder, self.embed, self.pos = decoder, embed, pos
        self.projection_layer = projection_layer

    def forward(self, x, tgt_mask=None):
        return self.projection_layer(self.decoder(self.pos(self.embed(x)), tgt_mask))


def build_decoder_only(vocab_size, seq_len, d_model=512, N=6, h=8, dropout=0.1,
                       d_ff=2048, tie_weights=False, scale_embeddings=True):
    embed = InputEmbeddings(d_model, vocab_size, scale_embeddings)
    pos = PositionalEncoding(d_model, seq_len, dropout)
    blocks = [
        DecoderOnlyBlock(d_model,
                         MultiHeadAttentionBlock(d_model, h, dropout),
                         FeedForwardBlock(d_model, d_ff, dropout),
                         dropout)
        for _ in range(N)
    ]
    decoder = DecoderOnlyDecoder(d_model, nn.ModuleList(blocks))
    model = DecoderOnly(decoder, embed, pos, ProjectionLayer(d_model, vocab_size))

    # FIX: GPT-2 style init instead of xavier_uniform_ on every dim>1 tensor.
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    model.apply(_init)

    # residual-output projections scaled by 1/sqrt(2N) so residual stream variance stays bounded
    for name, p in model.named_parameters():
        if name.endswith("w_o.weight") or name.endswith("linear_2.weight"):
            nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N))

    if tie_weights:
        model.projection_layer.proj.weight = model.embed.embedding.weight
    return model


def get_model(config, vocab_size):
    a = resolve_arch(config)
    return build_decoder_only(
        vocab_size, config["seq_len"],
        d_model=config["d_model"], N=a["n_layers"], h=config["n_heads"],
        dropout=config["dropout"], d_ff=a["d_ff"], tie_weights=a["tie_weights"],
    )

# %%
#5 Tokenizer + corpus packing
def build_or_load_tokenizer(config, texts_iter_fn):
    working_path = Path("/kaggle/working") / config["tokenizer_file"]
    if working_path.exists():
        print(f"[tokenizer] loading {working_path}")
        return Tokenizer.from_file(str(working_path))

    search_dir = config.get("tokenizer_source_dir") or config.get("resume_dataset_dir")
    if search_dir:
        d = Path(search_dir)
        cand = [d / config["tokenizer_file"]] + sorted(d.glob(Path(config["tokenizer_file"]).stem + "*.json"))
        for c in cand:
            if c.exists():
                print(f"[tokenizer] loading {c}")
                tok = Tokenizer.from_file(str(c))
                tok.save(str(working_path))
                return tok

    print(f"[tokenizer] building new '{config['tokenizer_mode']}' BPE, vocab={config['vocab_size']} ...")
    if config["tokenizer_mode"] == "bytelevel":
        
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=config["vocab_size"], min_frequency=2,
            special_tokens=["[PAD]", "[SOS]", "[EOS]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
    else:  
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=config["vocab_size"], min_frequency=2,
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        )
    tok.train_from_iterator(texts_iter_fn(), trainer=trainer)
    tok.save(str(working_path))
    print(f"[tokenizer] saved -> {working_path} (vocab={tok.get_vocab_size()})")
    return tok


def build_packed_corpus(config):
    
    cache = Path(config["cache_dir"]); cache.mkdir(parents=True, exist_ok=True)
    meta_p, train_p, val_p = cache / "meta.json", cache / "train.bin", cache / "val.bin"

    # reuse a cache uploaded as a Kaggle dataset from a previous session
    if not meta_p.exists() and config.get("resume_dataset_dir"):
        rd = Path(config["resume_dataset_dir"]) / "packed"
        if (rd / "meta.json").exists():
            print(f"[pack] reusing cached packed corpus from {rd}")
            cache = rd
            meta_p, train_p, val_p = cache / "meta.json", cache / "train.bin", cache / "val.bin"

    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta.get("vocab_size") == config["vocab_size"] and meta.get("tokenizer_mode") == config["tokenizer_mode"]:
            print(f"[pack] cache hit: {meta['train_tokens']:,} train / {meta['val_tokens']:,} val tokens")
            tok = Tokenizer.from_file(str(Path("/kaggle/working") / config["tokenizer_file"]))
            return str(train_p), str(val_p), tok, meta
        print("[pack] cache exists but config changed -> rebuilding")

    print(f"[pack] loading {config['datasource']}")
    ds_raw = load_dataset("json", data_files=config["datasource"], split="train")
    n = len(ds_raw)
    print(f"[pack] {n:,} documents")

    tok = build_or_load_tokenizer(
        config, lambda: (ds_raw[i]["text"] for i in range(min(n, 200_000))))
    eos_id = tok.token_to_id("[EOS]")
    assert eos_id is not None, "tokenizer has no [EOS]"

    rng = np.random.default_rng(config["split_seed"])
    perm = rng.permutation(n)
    n_val = max(1, int(config["val_fraction"] * n))
    val_idx, train_idx = set(perm[:n_val].tolist()), perm[n_val:]
    print(f"[pack] split: {len(train_idx):,} train docs / {n_val:,} val docs (seed={config['split_seed']})")

    def write_stream(indices, out_path, label):
        total_tokens, total_bytes = 0, 0
        CHUNK = 2000
        with open(out_path, "wb") as f:
            for s in tqdm(range(0, len(indices), CHUNK), desc=f"[pack] {label}"):
                idxs = list(indices[s:s + CHUNK])
                texts = [ds_raw[int(i)]["text"] for i in idxs]
                total_bytes += sum(len(t.encode("utf-8")) for t in texts)
                for enc in tok.encode_batch(texts):
                    ids = enc.ids + [eos_id]
                    total_tokens += len(ids)
                    np.asarray(ids, dtype=np.uint16).tofile(f)
        return total_tokens, total_bytes

    tr_tok, tr_bytes = write_stream(list(train_idx), train_p, "train")
    va_tok, va_bytes = write_stream(sorted(val_idx), val_p, "val")

    meta = {
        "vocab_size": config["vocab_size"],
        "tokenizer_mode": config["tokenizer_mode"],
        "train_tokens": tr_tok, "val_tokens": va_tok,
        "train_bytes": tr_bytes, "val_bytes": va_bytes,
        "val_tokens_per_byte": va_tok / max(1, va_bytes),
        "train_docs": len(train_idx), "val_docs": n_val,
    }
    meta_p.write_text(json.dumps(meta, indent=2))

    print(f"\n[pack] DONE. train {tr_tok:,} tokens | val {va_tok:,} tokens "
          f"| total {tr_tok + va_tok:,}")
    print(f"[pack] compression: {meta['val_tokens_per_byte']:.4f} tokens/byte "
          f"({1/meta['val_tokens_per_byte']:.2f} bytes/token)")
    return str(train_p), str(val_p), tok, meta


class PackedLMDataset(Dataset):
    """Contiguous windows over a packed token stream.
    Replaces the old per-document truncate+pad dataset."""
    def __init__(self, bin_path, seq_len, stride=None):
        self.bin_path, self.seq_len = bin_path, seq_len
        self.stride = stride or seq_len
        self.tokens = None                       # opened lazily, per DataLoader worker
        n_tok = os.path.getsize(bin_path) // 2   # uint16
        self.n = max(0, (n_tok - 1 - seq_len) // self.stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.tokens is None:
            self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        s = i * self.stride
        chunk = torch.from_numpy(self.tokens[s:s + self.seq_len + 1].astype(np.int64))
        return {"decoder_input": chunk[:-1], "label": chunk[1:]}


def get_ds(config):
    train_bin, val_bin, tok, meta = build_packed_corpus(config)
    train_ds = PackedLMDataset(train_bin, config["seq_len"])
    val_ds = PackedLMDataset(val_bin, config["seq_len"])
    print(f"[data] {len(train_ds):,} train windows | {len(val_ds):,} val windows "
          f"({len(train_ds) * config['seq_len']:,} train tokens per epoch)")

    common = dict(num_workers=2, pin_memory=True, persistent_workers=True)
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                          drop_last=True, **common)
    val_dl = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                        drop_last=False, **common)
    return train_dl, val_dl, tok, meta

# %%
# 6 Parameter count
def count_parameters_breakdown(model, tie_weights=False):
    c = lambda m: sum(p.numel() for p in m.parameters())
    proj = c(model.projection_layer)
    if tie_weights:
        proj -= model.embed.embedding.weight.numel()   
    breakdown = {
        "Input Embeddings": c(model.embed),
        "Positional Encoding": c(model.pos),
        "Decoder Layers (all N blocks)": c(model.decoder.layers),
        "Final LayerNorm": c(model.decoder.norm),
        "Projection Layer" + (" (tied — bias only)" if tie_weights else ""): proj,
    }
    total = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    emb = c(model.embed) + (0 if tie_weights else c(model.projection_layer))
    print(f"{'Component':<45}{'Params':>15}{'%':>9}")
    print("-" * 69)
    for k, v in breakdown.items():
        print(f"{k:<45}{v:>15,}{100*v/total:>8.1f}%")
    print("-" * 69)
    print(f"{'TOTAL':<45}{total:>15,}{100.0:>8.1f}%")
    print(f"{'  of which embedding/output':<45}{emb:>15,}{100*emb/total:>8.1f}%")
    print(f"{'  non-embedding (real capacity)':<45}{total-emb:>15,}{100*(total-emb)/total:>8.1f}%")
    return total


_cfg = get_config()
_a = resolve_arch(_cfg)
print(f"Preset: {_cfg['arch_preset']}  ->  n_layers={_a['n_layers']}, "
      f"d_ff={_a['d_ff']}, tie_weights={_a['tie_weights']}\n")
_m = get_model(_cfg, _cfg["vocab_size"])
count_parameters_breakdown(_m, _a["tie_weights"])
del _m; gc.collect()

# %%
# 7 Learning rate schedule
def lr_at_step(step, config, total_steps):
    peak, floor = config["lr"], config["lr"] * config["min_lr_ratio"]
    warmup = config["warmup_steps"]
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= total_steps:
        return floor
    prog = (step - warmup) / max(1, total_steps - warmup)
    return floor + 0.5 * (1.0 + math.cos(math.pi * prog)) * (peak - floor)


def make_param_groups(model, weight_decay):
    
    decay, no_decay, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.dim() >= 2 and "embedding" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    print(f"[optim] weight-decay tensors: {len(decay)} | no-decay tensors: {len(no_decay)}")
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]

# %%
# 8 Generation + validation
@torch.no_grad()
def generate_text(model, tokenizer, prompt_text, max_new_tokens=180, device="cuda",
                  temperature=0.2, top_k=7, top_p=0.92, repetition_penalty=1.2,
                  model_seq_len=512):
    net = model.module if hasattr(model, "module") else model
    net.eval()
    sos, eos = tokenizer.token_to_id("[SOS]"), tokenizer.token_to_id("[EOS]")
    ids = tokenizer.encode(prompt_text).ids if prompt_text else []
    if sos is not None:
        ids = [sos] + ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        ctx = x[:, -model_seq_len:]
        logits = net(ctx)[:, -1, :].float()

        if repetition_penalty != 1.0:
            for t in set(ctx[0].tolist()):
                logits[0, t] /= repetition_penalty if logits[0, t] > 0 else 1.0
                if logits[0, t] < 0:
                    logits[0, t] *= repetition_penalty

        logits = logits / max(1e-6, temperature)
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[logits < kth] = -float("inf")
        if top_p and top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone(); remove[..., 0] = False
            logits[0, si[0][remove[0]]] = -float("inf")

        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        if eos is not None and nxt.item() == eos:
            break
        x = torch.cat([x, nxt], dim=1)

    out = x[0].tolist()
    if sos is not None and out and out[0] == sos:
        out = out[1:]
    net.train()
    return tokenizer.decode(out)


@torch.no_grad()
def run_validation(model, val_dl, tokenizer, device, config, meta, print_msg=print,
                   do_generate=True):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot_loss, tot_tokens, nb = 0.0, 0, 0
    cap = config.get("val_max_batches") or len(val_dl)
    for batch in val_dl:
        x = batch["decoder_input"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with autocast():
            logits = model(x)
            loss = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        ntok = y.numel()
        tot_loss += loss.float().item() * ntok      # token-weighted, not batch-weighted
        tot_tokens += ntok
        nb += 1
        if nb >= cap:
            break

    avg = tot_loss / tot_tokens
    ppl = math.exp(min(20, avg))
    # tokenizer-independent: bits per UTF-8 byte
    bpb = avg * meta["val_tokens_per_byte"] / math.log(2)
    print_msg(f"Validation Loss: {avg:.4f} | Perplexity: {ppl:.4f} | "
              f"Bits/Byte: {bpb:.4f}   ({nb} batches, {tot_tokens:,} tokens)")

    if do_generate:
        try:
            p = "The history of"
            print_msg(f"TEST PROMPT: {p}")
            print_msg(f"GENERATED: {generate_text(model, tokenizer, p, device=device, model_seq_len=config['seq_len'])}")
            print_msg("-" * 80)
        except Exception as e:
            print_msg(f"Generation failed: {e}")

    model.train()
    return avg, ppl, bpb

# %%
#9 Training loop
def find_latest_checkpoint(config):
    dirs = [Path(get_working_dir(config))]
    if config.get("resume_dataset_dir"):
        dirs.append(Path(config["resume_dataset_dir"]))
    pat = re.compile(rf"^{re.escape(config['model_basename'])}(\d+)\.pt$")
    for d in dirs:
        if not d.exists():
            continue
        found = [(int(m.group(1)), f) for f in d.glob(f"{config['model_basename']}*.pt")
                 if (m := pat.match(f.name))]
        if found:
            found.sort(key=lambda t: t[0])          # numeric, not alphabetical
            print(f"[checkpoint] latest in {d}: {found[-1][1].name} (epoch {found[-1][0]})")
            return str(found[-1][1]), found[-1][0], str(d)
    print("[checkpoint] none found — starting from scratch.")
    return None, None, None


def load_history(config):
    default = {"epoch_numbers": [], "epoch_train_losses": [], "epoch_val_losses": [],
               "epoch_train_ppls": [], "epoch_val_ppls": [], "epoch_val_bpb": [],
               "lr_history_all": [], "grad_norm_history_all": [],
               "best_val_loss": float("inf"), "best_epoch": None}
    dirs = [Path(get_working_dir(config)), Path("/kaggle/working")]
    if config.get("resume_dataset_dir"):
        dirs.append(Path(config["resume_dataset_dir"]))
    stem = Path(config["history_filename"]).stem
    for d in dirs:
        if not d.exists():
            continue
        cands = [d / config["history_filename"]] + sorted(d.glob(f"{stem}*.json"))
        for c in cands:
            if c.exists():
                print(f"[history] loaded {c}")
                loaded = json.loads(c.read_text())
                for k, v in default.items():
                    loaded.setdefault(k, v)
                return loaded
    print("[history] none found — fresh history.")
    return default


def save_history(history, config):
    p = Path(get_working_dir(config)) / config["history_filename"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(history, indent=2))


def cleanup_old_checkpoints(folder, basename, keep):
    folder = Path(folder)
    pat = re.compile(rf"^{re.escape(basename)}(\d+)\.pt$")
    found = [(int(m.group(1)), f) for f in folder.glob(f"{basename}*.pt")
             if (m := pat.match(f.name))]
    found.sort(key=lambda t: t[0])
    for _, f in found[:-keep] if len(found) > keep else []:
        try:
            f.unlink(); print(f"[disk] removed {f.name}")
        except Exception:
            pass
    for f in folder.glob(f"{basename}*_batch*.pt"):
        try:
            f.unlink()
        except Exception:
            pass


def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wd = get_working_dir(config); Path(wd).mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    train_dl, val_dl, tokenizer, meta = get_ds(config)
    arch = resolve_arch(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    n_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    print(f"[model] preset={config['arch_preset']} | {n_params:,} params | dropout={config['dropout']}")

    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.AdamW(make_param_groups(model, config["weight_decay"]),
                                  lr=config["lr"], betas=(0.9, 0.95), eps=1e-8)
    scaler = GradScaler()

    accum = config["accumulate_grad_batches"]
    steps_per_epoch = len(train_dl) // accum
    total_steps = steps_per_epoch * config["total_epochs"]
    print(f"[sched] {steps_per_epoch:,} optim steps/epoch x {config['total_epochs']} epochs "
          f"= {total_steps:,} total | warmup={config['warmup_steps']} | "
          f"peak_lr={config['lr']} -> floor={config['lr']*config['min_lr_ratio']:.2e}")
    print(f"[sched] effective batch = {config['batch_size']*accum} seqs x {config['seq_len']} "
          f"= {config['batch_size']*accum*config['seq_len']:,} tokens/step")

    # ---- resume ----
    ckpt_path, ckpt_epoch, _ = find_latest_checkpoint(config)
    initial_epoch, global_step = 0, 0
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(state["model_state_dict"])
            initial_epoch = state["epoch"] + 1
            global_step = state.get("global_step", 0)
            if "optimizer_state_dict" in state:
                optimizer.load_state_dict(state["optimizer_state_dict"]); print("[resume] optimizer restored")
            if "scaler_state_dict" in state:
                scaler.load_state_dict(state["scaler_state_dict"]); print("[resume] GradScaler restored")
            print(f"[resume] epoch {initial_epoch}, global_step {global_step}")
        except RuntimeError as e:
            print(f"[resume] shape mismatch — starting fresh.\n  {e}")
            initial_epoch, global_step = 0, 0

    if torch.cuda.device_count() > 1:
        print(f"[gpu] DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    history = load_history(config)
    crit = nn.CrossEntropyLoss()

    patience = config["early_stopping_patience"]
    bad_epochs = 0
    if history["epoch_val_losses"]:
        for v in reversed(history["epoch_val_losses"]):
            if v <= history["best_val_loss"] + 1e-9:
                break
            bad_epochs += 1

    last_epoch = min(initial_epoch + config["epochs_this_session"], config["total_epochs"])
    if initial_epoch >= config["total_epochs"]:
        print(f"[done] already at epoch {initial_epoch} / {config['total_epochs']}.")
        return history
    print(f"\nTraining epochs {initial_epoch}..{last_epoch-1}\n")

    for epoch in range(initial_epoch, last_epoch):
        torch.cuda.empty_cache(); gc.collect()
        model.train()
        bar = tqdm(train_dl, desc=f"Epoch {epoch:02d}")
        optimizer.zero_grad(set_to_none=True)

        run_loss, run_tokens, win_loss, win_tok = 0.0, 0, 0.0, 0
        t0 = time.time()

        for i, batch in enumerate(bar):
            x = batch["decoder_input"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with autocast():
                logits = model(x)
                raw = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            scaler.scale(raw / accum).backward()

            ntok = y.numel()
            win_loss += raw.float().item() * ntok; win_tok += ntok
            run_loss += raw.float().item() * ntok; run_tokens += ntok

            if (i + 1) % accum == 0:
                lr_now = lr_at_step(global_step, config, total_steps)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

                gn = float(gnorm)
                history["lr_history_all"].append(lr_now)
                history["grad_norm_history_all"].append(gn)
                writer.add_scalar("train loss", win_loss / win_tok, global_step)
                writer.add_scalar("learning rate", lr_now, global_step)
                writer.add_scalar("grad norm", gn, global_step)
                bar.set_postfix({"loss": f"{win_loss/win_tok:6.3f}", "lr": f"{lr_now:.2e}"})
                win_loss, win_tok = 0.0, 0
                global_step += 1

        avg_train = run_loss / max(1, run_tokens)
        tok_s = run_tokens / (time.time() - t0)
        print(f"[epoch {epoch}] train loss {avg_train:.4f} | {tok_s:,.0f} tok/s | "
              f"{run_tokens:,} tokens seen")

        avg_val, val_ppl, val_bpb = run_validation(
            model, val_dl, tokenizer, device, config, meta, print_msg=bar.write)

        history["epoch_numbers"].append(epoch)
        history["epoch_train_losses"].append(avg_train)
        history["epoch_val_losses"].append(avg_val)
        history["epoch_train_ppls"].append(math.exp(min(20, avg_train)))
        history["epoch_val_ppls"].append(val_ppl)
        history["epoch_val_bpb"].append(val_bpb)

        to_save = model.module if hasattr(model, "module") else model
        payload = lambda: {"epoch": epoch, "model_state_dict": to_save.state_dict(),
                           "optimizer_state_dict": optimizer.state_dict(),
                           "scaler_state_dict": scaler.state_dict(),
                           "global_step": global_step, "val_loss": avg_val,
                           "config": {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool, type(None)))}}

        if avg_val < history["best_val_loss"]:
            history["best_val_loss"], history["best_epoch"] = avg_val, epoch
            bad_epochs = 0
            torch.save(payload(), f"{wd}/{config['model_basename']}best.pt")
            print(f"[best] new best val loss {avg_val:.4f} at epoch {epoch}")
        else:
            bad_epochs += 1
            print(f"[best] no improvement for {bad_epochs} epoch(s) "
                  f"(best={history['best_val_loss']:.4f} @ epoch {history['best_epoch']})")

        torch.save(payload(), f"{wd}/{config['model_basename']}{epoch}.pt")
        cleanup_old_checkpoints(wd, config["model_basename"], config["keep_last_n_checkpoints"])
        save_history(history, config)
        torch.cuda.empty_cache(); gc.collect()

        if config["use_early_stopping"] and bad_epochs >= patience:
            print(f"[early-stop] no val improvement for {patience} epochs. Stopping at epoch {epoch}.")
            break

    print(f"\nSession complete. Best val loss {history['best_val_loss']:.4f} "
          f"at epoch {history['best_epoch']}")
    print("\n>>> DOWNLOAD before ending the session: <<<")
    print(f"  1. {wd}/                      (checkpoints + history)")
    print(f"  2. /kaggle/working/{config['tokenizer_file']}")
    print(f"  3. {config['cache_dir']}/     (packed corpus — saves ~20 min next session)")
    return history

# %%
# 10 Run
if __name__ == "__main__":
    config = get_config()
    history = train_model(config)

    ep = [e + 1 for e in history["epoch_numbers"]]
    if ep:
        fig, ax = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"Pretraining Diagnostics — {config['arch_preset']} (through epoch {ep[-1]})",
                     fontsize=14)

        ax[0, 0].plot(ep, history["epoch_train_losses"], "o-", color="tab:blue", label="Train")
        ax[0, 0].plot(ep, history["epoch_val_losses"], "s-", color="tab:red", label="Val")
        ax[0, 0].axhline(3.0, ls="--", c="gray", lw=1, label="target 3.0")
        ax[0, 0].set_title("Loss"); ax[0, 0].set_xlabel("Epoch"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)

        ax[0, 1].plot(ep, history["epoch_val_ppls"], "s-", color="tab:red", label="Val PPL")
        ax[0, 1].set_title("Perplexity"); ax[0, 1].set_xlabel("Epoch"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)

        ax[1, 0].plot(history["lr_history_all"], color="tab:green")
        ax[1, 0].set_title("Learning Rate"); ax[1, 0].set_xlabel("Optimizer Step"); ax[1, 0].grid(alpha=.3)

        ax[1, 1].plot(history["grad_norm_history_all"], color="tab:purple", lw=.6)
        ax[1, 1].set_title("Gradient Norm"); ax[1, 1].set_xlabel("Optimizer Step"); ax[1, 1].grid(alpha=.3)

        plt.tight_layout()
        plt.savefig("/kaggle/working/pretraining_diagnostics.png", dpi=150, bbox_inches="tight")
        plt.show()

        if history.get("epoch_val_bpb"):
            print("\nEpoch | Train Loss | Val Loss |  Val PPL | Bits/Byte")
            print("-" * 56)
            for i, e in enumerate(ep):
                print(f"{e:5d} | {history['epoch_train_losses'][i]:10.4f} | "
                      f"{history['epoch_val_losses'][i]:8.4f} | "
                      f"{history['epoch_val_ppls'][i]:8.2f} | {history['epoch_val_bpb'][i]:9.4f}")

# %%
# 11 
config = get_config()
train_bin, val_bin, tok, meta = build_packed_corpus(config)

total = meta["train_tokens"] + meta["val_tokens"]
docs = meta["train_docs"] + meta["val_docs"]
old_effective = docs * (config["seq_len"] - 1)   # what the truncating dataset actually saw

print(f"\n{'':-<62}")
print(f"{'Documents in corpus':<40}{docs:>20,}")
print(f"{'Total tokens in corpus':<40}{total:>20,}")
print(f"{'Avg tokens/document':<40}{total/docs:>20,.0f}")
print(f"{'':-<62}")
print(f"{'OLD pipeline (truncate to 511)':<40}{min(old_effective,total):>20,}")
print(f"{'NEW pipeline (packed)':<40}{total:>20,}")
print(f"{'Corpus utilisation, old':<40}{100*min(old_effective,total)/total:>19.1f}%")
print(f"{'Corpus utilisation, new':<40}{100.0:>19.1f}%")
print(f"{'Effective data multiplier':<40}{total/max(1,min(old_effective,total)):>19.2f}x")
print(f"{'':-<62}")

a = resolve_arch(config)
n_params = 49_915_184 if not a["tie_weights"] else 49_944_528
print(f"{'Tokens per parameter, old':<40}{min(old_effective,total)/n_params:>20.2f}")
print(f"{'Tokens per parameter, new':<40}{total/n_params:>20.2f}")

# %%
# 12 inference from a checkpoint
config = get_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
_, _, tokenizer, meta = get_ds(config)
model = get_model(config, tokenizer.get_vocab_size()).to(device)

best = Path(get_working_dir(config)) / f"{config['model_basename']}best.pt"
path, ep, _ = (str(best), "best", None) if best.exists() else find_latest_checkpoint(config)
if path:
    st = torch.load(path, map_location=device)
    model.load_state_dict(st["model_state_dict"])
    print(f"Loaded {path} (epoch {st.get('epoch')}, val_loss={st.get('val_loss')})")
else:
    print("No checkpoint found — model is randomly initialised.")

for p in ["The history of", "In a 5G network, the base station",
          "Orthogonal frequency-division multiplexing is"]:
    print("=" * 80)
    print(f"PROMPT: {p}")
    print(generate_text(model, tokenizer, p, device=device,
                        model_seq_len=config["seq_len"], max_new_tokens=150))
