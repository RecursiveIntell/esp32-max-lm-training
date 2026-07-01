#!/usr/bin/env python3
"""
Train a max-practical ESP32-S3 char-level LSTM on the MSI GTX 1070.

Target: 3-layer hidden=512 LSTM, ~6.34M params, int8 export ~6.1 MiB plus
small f32 biases. This is near the practical 8MB PSRAM ceiling while leaving
room for activations/firmware/runtime buffers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
VOCAB_CHARS = list("abcdefghijklmnopqrstuvwxyz .,!?'\n")
VOCAB_SIZE = len(VOCAB_CHARS)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB_CHARS)}

EDGE_AI_TEMPLATES = [
    "the esp32 reads the room and sends the evidence to the local model. ",
    "the sensor sees warm air and heavy humidity. the gateway decides what matters. ",
    "a tiny board watches the world. a larger model answers only when needed. ",
    "the oled shows a short answer. the receipt keeps the reason. ",
    "local first means the room can think without the cloud. ",
    "the temperature is evidence. the humidity is evidence. the action must match the evidence. ",
    "when the data is stale the model must say the data is stale. ",
    "the small model stays awake. the big model wakes only when the signal is worth it. ",
    "the gateway writes a receipt before it takes action. ",
    "do not invent readings. do not pretend the room is safe if the sensor is missing. ",
    "the physical node is cheap, quiet, and always watching. ",
    "the msi runs gemma. the esp32 sends sensor context. the oled shows the result. ",
]

class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden
        self.num_layers = layers
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, vocab_size)
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                n = p.shape[0]
                p.data[n // 4:n // 2].fill_(1.0)  # forget gate bias
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(self.embed(x), hidden)
        return self.fc(out), hidden
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

def filter_text(text: str) -> str:
    text = text.lower()
    out = []
    for ch in text:
        if ch in CHAR_TO_IDX:
            out.append(ch)
        elif ch in "\r\t":
            out.append(" ")
        elif ch == "\n":
            out.append("\n")
    return "".join(out)

def ensure_corpus(data_dir: Path, max_chars: int, edge_chars: int) -> str:
    data_dir.mkdir(parents=True, exist_ok=True)
    tiny_path = data_dir / "TinyStories-train.txt"
    corpus_path = data_dir / f"corpus_{max_chars}_{edge_chars}.txt"
    if corpus_path.exists() and corpus_path.stat().st_size > 1_000_000:
        return corpus_path.read_text(encoding="utf-8")
    if not tiny_path.exists() or tiny_path.stat().st_size < 1_000_000:
        print(f"Downloading TinyStories to {tiny_path}", flush=True)
        urllib.request.urlretrieve(TINYSTORIES_URL, tiny_path)
    with tiny_path.open("r", encoding="utf-8") as f:
        tiny = f.read(max_chars)
    tiny = filter_text(tiny)
    rng = random.Random(42)
    edge = []
    while sum(len(x) for x in edge) < edge_chars:
        edge.append(rng.choice(EDGE_AI_TEMPLATES))
        if rng.random() < 0.3:
            edge.append("once upon a time there was a tiny model that lived on a sensor board. ")
    edge = filter_text("".join(edge))[:edge_chars]
    # Interleave edge-domain text so it survives training instead of being tail-only.
    chunks = []
    ti = ei = 0
    while ti < len(tiny) or ei < len(edge):
        if ti < len(tiny):
            chunks.append(tiny[ti:ti+2500]); ti += 2500
        if ei < len(edge):
            chunks.append(edge[ei:ei+700]); ei += 700
    corpus = "".join(chunks)
    corpus_path.write_text(corpus, encoding="utf-8")
    return corpus

def encode(text: str) -> np.ndarray:
    return np.array([CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX], dtype=np.int64)

def batch_iter(data: np.ndarray, seq_len: int, batch_size: int):
    n = len(data) // (seq_len * batch_size)
    data = data[:n * seq_len * batch_size].reshape(batch_size, -1)
    for i in range(0, data.shape[1] - seq_len - 1, seq_len):
        yield data[:, i:i+seq_len], data[:, i+1:i+seq_len+1]

def lr_at(step, warmup, lr, min_lr, total):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))

def evaluate(model, data, device, seq_len, max_chunks=512):
    model.eval(); losses=[]
    with torch.no_grad():
        stride = max(seq_len, (len(data) - seq_len - 1) // max_chunks)
        for i in range(0, len(data) - seq_len - 1, stride):
            x = torch.from_numpy(data[i:i+seq_len]).unsqueeze(0).to(device)
            y = torch.from_numpy(data[i+1:i+seq_len+1]).unsqueeze(0).to(device)
            logits, _ = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)).item())
            if len(losses) >= max_chunks:
                break
    model.train()
    return float(np.mean(losses)) if losses else float("inf")

def sample(model, seed, length, device, temp=0.5):
    model.eval()
    chars = [CHAR_TO_IDX[c] for c in seed.lower() if c in CHAR_TO_IDX] or [CHAR_TO_IDX['i']]
    hidden = None
    result = seed
    with torch.no_grad():
        for ch in chars[:-1]:
            _, hidden = model(torch.tensor([[ch]], device=device), hidden)
        cur = chars[-1]
        for _ in range(length):
            logits, hidden = model(torch.tensor([[cur]], device=device), hidden)
            probs = F.softmax(logits[0, -1] / temp, dim=-1)
            cur = torch.multinomial(probs, 1).item()
            result += IDX_TO_CHAR[cur]
    model.train()
    return result

def q8(arr: np.ndarray):
    m = float(np.max(np.abs(arr)))
    scale = m / 127.0 if m > 0 else 1.0
    return np.round(arr / scale).clip(-128, 127).astype(np.int8), scale

def export_weights(model, out_path: Path):
    st = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    hidden = model.hidden_size; layers = model.num_layers
    lines = [
        "// Auto-generated max-practical ESP32-S3 char LSTM weights",
        f"// params: {model.count_params():,}; int8 weights + f32 biases",
        f"pub const VOCAB_SIZE: usize = {VOCAB_SIZE};",
        f"pub const HIDDEN: usize = {hidden};",
        f"pub const NUM_LAYERS: usize = {layers};",
        f"pub const VOCAB: [u8; VOCAB_SIZE] = {[ord(c) for c in VOCAB_CHARS]};",
        "",
    ]
    def emit_i8(name, arr):
        flat = arr.reshape(-1)
        lines.append(f"pub const {name}: [i8; {flat.size}] = [")
        for i in range(0, flat.size, 32):
            lines.append("    " + ", ".join(str(int(x)) for x in flat[i:i+32]) + ",")
        lines.append("];\n")
    def emit_f32(name, arr):
        flat = arr.reshape(-1)
        lines.append(f"pub const {name}: [f32; {flat.size}] = [")
        for i in range(0, flat.size, 8):
            lines.append("    " + ", ".join(f"{float(x):.7e}" for x in flat[i:i+8]) + ",")
        lines.append("];\n")
    q, s = q8(st['embed.weight']); lines.append(f"pub const EMBED_SCALE: f32 = {s:.7e};"); emit_i8('EMBED', q)
    for layer in range(layers):
        q, s = q8(st[f'lstm.weight_ih_l{layer}']); lines.append(f"pub const WIH_L{layer}_SCALE: f32 = {s:.7e};"); emit_i8(f'WIH_L{layer}', q)
        q, s = q8(st[f'lstm.weight_hh_l{layer}']); lines.append(f"pub const WHH_L{layer}_SCALE: f32 = {s:.7e};"); emit_i8(f'WHH_L{layer}', q)
        emit_f32(f'BIAS_L{layer}', st[f'lstm.bias_ih_l{layer}'] + st[f'lstm.bias_hh_l{layer}'])
    q, s = q8(st['fc.weight']); lines.append(f"pub const FC_W_SCALE: f32 = {s:.7e};"); emit_i8('FC_W', q)
    emit_f32('FC_B', st['fc.bias'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--seq-len', type=int, default=64)
    ap.add_argument('--batch-size', type=int, default=96)
    ap.add_argument('--lr', type=float, default=1.5e-3)
    ap.add_argument('--min-lr', type=float, default=1e-5)
    ap.add_argument('--warmup', type=int, default=150)
    ap.add_argument('--max-corpus-chars', type=int, default=8_000_000)
    ap.add_argument('--edge-chars', type=int, default=300_000)
    ap.add_argument('--out', default='runs/esp32s3_max_lstm_h512_l3')
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
    print(json.dumps({'device': device, 'cuda': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'args': vars(args)}, indent=2), flush=True)

    corpus = ensure_corpus(Path('data'), args.max_corpus_chars, args.edge_chars)
    data = encode(corpus)
    val_n = max(10_000, int(len(data) * 0.03))
    train = data[:-val_n]; val = data[-val_n:]
    print(f"corpus tokens={len(data):,} train={len(train):,} val={len(val):,} vocab={VOCAB_SIZE}", flush=True)

    model = CharLSTM(VOCAB_SIZE, args.hidden, args.layers, dropout=0.1).to(device)
    print(f"params={model.count_params():,} int8_weight_budget≈{model.count_params()/1024/1024:.2f} MiB", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    best_loss = float('inf'); best_path = out / 'best.pt'; start = time.time(); step = 0; epoch = 0
    log = []
    while step < args.steps:
        epoch += 1
        offset = random.randint(0, args.seq_len - 1) if epoch > 1 else 0
        epoch_data = np.concatenate([train[offset:], train[:offset]])
        for xb, yb in batch_iter(epoch_data, args.seq_len, args.batch_size):
            if step >= args.steps: break
            lr = lr_at(step, args.warmup, args.lr, args.min_lr, args.steps)
            for g in opt.param_groups: g['lr'] = lr
            x = torch.from_numpy(xb).to(device, non_blocking=True)
            y = torch.from_numpy(yb).to(device, non_blocking=True)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % 50 == 0:
                elapsed = time.time() - start
                print(f"step {step:05d}/{args.steps} loss={loss.item():.4f} ppl={math.exp(loss.item()):.2f} lr={lr:.2e} elapsed={elapsed/60:.1f}m", flush=True)
            if step > 0 and step % 250 == 0:
                vl = evaluate(model, val, device, args.seq_len)
                vp = math.exp(vl)
                print(f"VAL step={step} loss={vl:.4f} ppl={vp:.2f}", flush=True)
                log.append({'step': step, 'train_loss': float(loss.item()), 'val_loss': vl, 'val_ppl': vp, 'elapsed_s': time.time()-start})
                if vl < best_loss:
                    best_loss = vl
                    torch.save({'model_state': model.state_dict(), 'args': vars(args), 'vocab_chars': VOCAB_CHARS, 'param_count': model.count_params(), 'best_val_loss': best_loss}, best_path)
                    print(f"BEST saved {best_path}", flush=True)
                print('SAMPLE:', sample(model, 'the sensor says ', 180, device, temp=0.55)[:240].replace('\n','\\n'), flush=True)
                (out / 'training_log.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
            step += 1
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
    final_loss = evaluate(model, val, device, args.seq_len)
    torch.save({'model_state': model.state_dict(), 'args': vars(args), 'vocab_chars': VOCAB_CHARS, 'param_count': model.count_params(), 'best_val_loss': min(best_loss, final_loss), 'final_val_loss': final_loss}, out / 'final.pt')
    export_weights(model, out / 'weights' / 'esp32s3_max_lstm_weights.rs')
    samples=[]
    for seed in ['the sensor says ', 'the room feels ', 'the esp32 sees ']:
        for temp in [0.35, 0.55, 0.75]:
            samples.append(f"=== {seed!r} temp={temp} ===\n{sample(model, seed, 300, device, temp)}\n")
    (out / 'samples.txt').write_text('\n'.join(samples), encoding='utf-8')
    summary={'params': model.count_params(), 'hidden': args.hidden, 'layers': args.layers, 'final_val_loss': final_loss, 'final_val_ppl': math.exp(final_loss), 'best_val_loss': best_loss, 'elapsed_s': time.time()-start, 'weights': str(out / 'weights' / 'esp32s3_max_lstm_weights.rs')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('SUMMARY', json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
