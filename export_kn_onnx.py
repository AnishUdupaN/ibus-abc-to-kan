import json
import os
import sqlite3
import sys
import types
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

urduhack_stub = types.ModuleType("urduhack")
urduhack_stub.normalize = lambda text: text
sys.modules["urduhack"] = urduhack_stub

_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

import ai4bharat.transliteration as aib_pkg
from ai4bharat.transliteration import XlitEngine

OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path.home() / ".local/share/indicxlit-kn/models")))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_transformer(obj, visited=None):
    if visited is None:
        visited = set()
    if id(obj) in visited or obj is None:
        return None
    visited.add(id(obj))

    if hasattr(obj, "encoder") and hasattr(obj, "decoder") and obj.encoder is not None:
        return obj
    if isinstance(obj, (list, tuple, torch.nn.ModuleList)):
        for item in obj:
            found = find_transformer(item, visited)
            if found is not None:
                return found
    if hasattr(obj, "__dict__"):
        for val in obj.__dict__.values():
            found = find_transformer(val, visited)
            if found is not None:
                return found
    return None


print("1/5 Loading IndicXlit Kannada model...")
xlit = XlitEngine("kn", beam_width=4, rescore=True)
base = xlit.transliterator

model = find_transformer(xlit)
assert model is not None, "Could not locate Transformer model inside XlitEngine"
model.eval()

src_dict = base.task.source_dictionary
tgt_dict = base.task.target_dictionary
sa0 = model.decoder.layers[0].self_attn
num_layers = len(model.decoder.layers)
num_heads = int(sa0.num_heads)
head_dim = int(sa0.head_dim)
embed_dim = int(sa0.embed_dim)

with open(OUT_DIR / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "src_symbols": src_dict.symbols,
            "tgt_symbols": tgt_dict.symbols,
            "pad_idx": int(src_dict.pad()),
            "eos_idx": int(src_dict.eos()),
            "unk_idx": int(src_dict.unk()),
            "kn_token": "__kn__",
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        f,
        ensure_ascii=False,
    )

print("2/5 Building sorted Kannada SQLite database...")
kn_probs = {}
pkg_root = Path(aib_pkg.__file__).resolve().parent

for candidate_file in pkg_root.rglob("*kn*.json"):
    with open(candidate_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)
        if isinstance(data, dict) and len(data) > 1000:
            kn_probs = data.get("kn", data)
            print(f"    Found Kannada dict in {candidate_file.name} ({len(kn_probs)} words)")
            break

if not kn_probs:
    for dict_file in pkg_root.rglob("word_prob_dicts*.json"):
        with open(dict_file, "r", encoding="utf-8") as fp:
            kn_probs = json.load(fp).get("kn", {})
        break

db_path = OUT_DIR / "kn_prob.db"
if db_path.exists():
    db_path.unlink()

sorted_items = sorted(kn_probs.items(), key=lambda kv: kv[0])
del kn_probs

conn = sqlite3.connect(db_path)
conn.execute("PRAGMA journal_mode = OFF")
conn.execute("PRAGMA synchronous = OFF")
conn.execute("PRAGMA cache_size = -65536")
conn.execute("CREATE TABLE probs (word TEXT PRIMARY KEY, prob REAL) WITHOUT ROWID")
conn.executemany("INSERT INTO probs (word, prob) VALUES (?, ?)", sorted_items)
conn.commit()
conn.close()
del sorted_items
print(f"    Saved SQLite DB to {db_path}")


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, src_tokens):
        out = self.encoder(src_tokens)
        return out["encoder_out"][0]


class FastStepDecoder(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.layers = decoder.layers

        _ = decoder.embed_positions(torch.tensor([[2, 4]], dtype=torch.long))
        pos_mod = decoder.embed_positions
        if hasattr(pos_mod, "weights") and pos_mod.weights is not None:
            pos_table = pos_mod.weights.detach().clone().float()
        elif hasattr(pos_mod, "weight"):
            pos_table = pos_mod.weight.detach().clone().float()
        else:
            raise RuntimeError("Unsupported positional embedding module")

        self.register_buffer("pos_weights", pos_table)
        self.pad_idx = int(getattr(pos_mod, "padding_idx", 1))

    def forward(self, last_token, step_idx, encoder_out, past_keys, past_values):
        x = self.decoder.embed_scale * self.decoder.embed_tokens(last_token)
        pos_idx = step_idx + (self.pad_idx + 1)
        pos_emb = self.pos_weights.index_select(0, pos_idx.view(-1)).view(1, 1, -1)
        x = x + pos_emb

        if getattr(self.decoder, "project_in_dim", None) is not None:
            x = self.decoder.project_in_dim(x)
        if getattr(self.decoder, "layernorm_embedding", None) is not None:
            x = self.decoder.layernorm_embedding(x)

        x = x.transpose(0, 1)
        new_keys = []
        new_values = []

        for i, layer in enumerate(self.layers):
            residual = x
            if layer.normalize_before:
                x = layer.self_attn_layer_norm(x)

            sa = layer.self_attn
            q = (sa.q_proj(x) * sa.scaling).reshape(1, -1, sa.head_dim).transpose(0, 1)
            k = sa.k_proj(x).reshape(1, -1, sa.head_dim).transpose(0, 1)
            v = sa.v_proj(x).reshape(1, -1, sa.head_dim).transpose(0, 1)

            k_cat = torch.cat([past_keys[i], k], dim=1)
            v_cat = torch.cat([past_values[i], v], dim=1)
            new_keys.append(k_cat)
            new_values.append(v_cat)

            attn_weights = torch.bmm(q, k_cat.transpose(1, 2))
            attn_probs = torch.softmax(attn_weights, dim=-1)
            attn = torch.bmm(attn_probs, v_cat).transpose(0, 1).reshape(1, -1, sa.embed_dim)
            x = residual + sa.out_proj(attn)
            if not layer.normalize_before:
                x = layer.self_attn_layer_norm(x)

            if layer.encoder_attn is not None:
                residual = x
                if layer.normalize_before:
                    x = layer.encoder_attn_layer_norm(x)

                ca = layer.encoder_attn
                src_len = encoder_out.size(0)
                cq = (ca.q_proj(x) * ca.scaling).reshape(1, -1, ca.head_dim).transpose(0, 1)
                ck = ca.k_proj(encoder_out).reshape(src_len, -1, ca.head_dim).transpose(0, 1)
                cv = ca.v_proj(encoder_out).reshape(src_len, -1, ca.head_dim).transpose(0, 1)

                c_weights = torch.bmm(cq, ck.transpose(1, 2))
                c_probs = torch.softmax(c_weights, dim=-1)
                c_attn = torch.bmm(c_probs, cv).transpose(0, 1).reshape(1, -1, ca.embed_dim)
                x = residual + ca.out_proj(c_attn)
                if not layer.normalize_before:
                    x = layer.encoder_attn_layer_norm(x)

            residual = x
            if layer.normalize_before:
                x = layer.final_layer_norm(x)
            x = layer.activation_fn(layer.fc1(x))
            x = residual + layer.fc2(x)
            if not layer.normalize_before:
                x = layer.final_layer_norm(x)

        if getattr(self.decoder, "layer_norm", None) is not None:
            x = self.decoder.layer_norm(x)

        x = x.transpose(0, 1)
        if getattr(self.decoder, "project_out_dim", None) is not None:
            x = self.decoder.project_out_dim(x)

        logits = self.decoder.output_layer(x)[:, 0, :]
        return (
            torch.log_softmax(logits, dim=-1),
            torch.stack(new_keys, dim=0),
            torch.stack(new_values, dim=0),
        )


print("3/5 Tracing Encoder & KV-Cached Decoder to FP32 ONNX...")
enc_wrap = EncoderWrapper(model.encoder).eval()
dec_wrap = FastStepDecoder(model.decoder).eval()

enc_fp32 = OUT_DIR / "encoder_fp32.onnx"
dec_fp32 = OUT_DIR / "decoder_step_fp32.onnx"

dummy_src = torch.tensor([[4, 5, 6, 7, 2]], dtype=torch.long)
dummy_tok = torch.tensor([[2], [2]], dtype=torch.long)
dummy_step = torch.tensor([3], dtype=torch.long)
dummy_enc = torch.randn(6, 2, embed_dim)
dummy_pk = torch.randn(num_layers, 2 * num_heads, 3, head_dim)
dummy_pv = torch.randn(num_layers, 2 * num_heads, 3, head_dim)

with torch.no_grad():
    torch.onnx.export(
        enc_wrap,
        (dummy_src,),
        str(enc_fp32),
        input_names=["src_tokens"],
        output_names=["encoder_out"],
        dynamic_axes={
            "src_tokens": {0: "batch", 1: "src_len"},
            "encoder_out": {0: "src_len", 1: "batch"},
        },
        opset_version=14,
        dynamo=False,
    )

    torch.onnx.export(
        dec_wrap,
        (dummy_tok, dummy_step, dummy_enc, dummy_pk, dummy_pv),
        str(dec_fp32),
        input_names=["last_token", "step_idx", "encoder_out", "past_keys", "past_values"],
        output_names=["log_probs", "present_keys", "present_values"],
        dynamic_axes={
            "last_token": {0: "batch"},
            "encoder_out": {0: "src_len", 1: "batch"},
            "past_keys": {1: "batch_heads", 2: "past_len"},
            "past_values": {1: "batch_heads", 2: "past_len"},
            "log_probs": {0: "batch"},
            "present_keys": {1: "batch_heads", 2: "new_len"},
            "present_values": {1: "batch_heads", 2: "new_len"},
        },
        opset_version=14,
        dynamo=False,
    )

print("4/5 Quantizing ONNX models to INT8...")
quantize_dynamic(str(enc_fp32), str(OUT_DIR / "encoder.int8.onnx"), weight_type=QuantType.QInt8)
quantize_dynamic(str(dec_fp32), str(OUT_DIR / "decoder_step.int8.onnx"), weight_type=QuantType.QInt8)
enc_fp32.unlink()
dec_fp32.unlink()

print("5/5 Running quick ONNX graph sanity check...")
enc_s = ort.InferenceSession(str(OUT_DIR / "encoder.int8.onnx"), providers=["CPUExecutionProvider"])
dec_s = ort.InferenceSession(str(OUT_DIR / "decoder_step.int8.onnx"), providers=["CPUExecutionProvider"])
src_to_id = {s: i for i, s in enumerate(src_dict.symbols)}
ids = np.array(
    [[src_to_id["__kn__"]] + [src_to_id.get(c, src_dict.unk()) for c in "anna"] + [src_dict.eos()]],
    dtype=np.int64,
)
eo = enc_s.run(None, {"src_tokens": ids})[0]
pk = np.zeros((num_layers, num_heads, 0, head_dim), dtype=np.float32)
pv = np.zeros((num_layers, num_heads, 0, head_dim), dtype=np.float32)
tok = np.array([[src_dict.eos()]], dtype=np.int64)
out_chars = []
for step in range(10):
    lp, pk, pv = dec_s.run(
        None,
        {
            "last_token": tok,
            "step_idx": np.array([step], dtype=np.int64),
            "encoder_out": eo,
            "past_keys": pk,
            "past_values": pv,
        },
    )
    nxt = int(np.argmax(lp[0]))
    if nxt == src_dict.eos():
        break
    if nxt not in (0, 1, 2, 3):
        out_chars.append(tgt_dict.symbols[nxt])
    tok = np.array([[nxt]], dtype=np.int64)
print(f"    Sanity check 'anna' (greedy) -> {''.join(out_chars)}")
print(f"Done! Bundle saved to: {OUT_DIR}")
