import ctypes
import gc
import json
import sqlite3
from functools import lru_cache
from pathlib import Path
import numpy as np
import onnxruntime as ort

BASE_DIR = Path.home() / ".local/share/indicxlit-kn"
MODEL_DIR = BASE_DIR / "models"
PREFS_FILE = BASE_DIR / "user_prefs.json"

# Spoken Kannada geminate/anusvara -> mixed retroflex+dental cluster expansions
MIXED_CLUSTER_RULES = (
    ("ಡ್ಡ", "ಡ್ದ"),
    ("ದ್ದ", "ಡ್ದ"),
    ("ಟ್ಟ", "ಟ್ತ"),
    ("ತ್ತ", "ಟ್ತ"),
    ("ಂದ", "ಣ್ದ"),
    ("ಂಡ", "ಣ್ದ"),
    ("ನ್ದ", "ಣ್ದ"),
)

# Retroflex counterparts for standalone/short syllables
SHORT_SYLLABLE_EXTRAS: dict[str, tuple[str, ...]] = {
    "n": ("ಣ್",), "na": ("ಣ",), "naa": ("ಣಾ",), "ni": ("ಣಿ",),
    "nee": ("ಣೀ",), "nu": ("ಣು",), "noo": ("ಣೂ",), "ne": ("ಣೆ", "ಣೇ"),
    "nai": ("ಣೈ",), "no": ("ಣೊ", "ಣೋ"), "nau": ("ಣೌ",),
    "l": ("ಳ್",), "la": ("ಳ",), "laa": ("ಳಾ",), "li": ("ಳಿ",),
    "lee": ("ಳೀ",), "lu": ("ಳು",), "loo": ("ಳೂ",), "le": ("ಳೆ", "ಳೇ"),
    "lai": ("ಳೈ",), "lo": ("ಳೊ", "ಳೋ"), "lau": ("ಳೌ",),
}

# Explicit Shift-key overrides (e.g., Na -> ಣ at slot #1 immediately)
EXACT_CASE_OVERRIDES: dict[str, str] = {
    "N": "ಣ್", "Na": "ಣ", "Naa": "ಣಾ", "Ni": "ಣಿ", "Nee": "ಣೀ",
    "Nu": "ಣು", "Noo": "ಣೂ", "Ne": "ಣೆ", "NE": "ಣೇ", "Nai": "ಣೈ",
    "No": "ಣೊ", "NO": "ಣೋ", "Nau": "ಣೌ",
    "L": "ಳ್", "La": "ಳ", "Laa": "ಳಾ", "Li": "ಳಿ", "Lee": "ಳೀ",
    "Lu": "ಳು", "Loo": "ಳೂ", "Le": "ಳೆ", "LE": "ಳೇ", "Lai": "ಳೈ",
    "Lo": "ಳೊ", "LO": "ಳೋ", "Lau": "ಳೌ",
}


def _trim_heap() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class KannadaOnnxTransliterator:
    def __init__(self, model_dir: Path = MODEL_DIR):
        with open(model_dir / "vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.src_symbols = vocab["src_symbols"]
        self.tgt_symbols = vocab["tgt_symbols"]
        self.src_to_id = {s: i for i, s in enumerate(self.src_symbols)}
        self.eos_idx = vocab["eos_idx"]
        self.unk_idx = vocab["unk_idx"]
        self.kn_token_id = self.src_to_id.get(vocab["kn_token"], self.unk_idx)

        self.num_layers = vocab["num_layers"]
        self.num_heads = vocab["num_heads"]
        self.head_dim = vocab["head_dim"]

        self.user_prefs: dict[str, str] = {}
        if PREFS_FILE.exists():
            try:
                with open(PREFS_FILE, "r", encoding="utf-8") as f:
                    self.user_prefs = json.load(f)
            except Exception:
                self.user_prefs = {}

        db_uri = f"file:{model_dir / 'kn_prob.db'}?mode=ro&immutable=1"
        self.db = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
        self.db.execute("PRAGMA cache_size = -512")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.enable_mem_reuse = True
        opts.add_session_config_entry("session.disable_prepacking", "1")

        self.enc_sess = ort.InferenceSession(
            str(model_dir / "encoder.int8.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.dec_sess = ort.InferenceSession(
            str(model_dir / "decoder_step.int8.onnx"), opts, providers=["CPUExecutionProvider"]
        )

        _trim_heap()

    def _encode_word(self, word: str) -> np.ndarray:
        ids = (
            [self.kn_token_id]
            + [self.src_to_id.get(ch, self.unk_idx) for ch in word.lower()]
            + [self.eos_idx]
        )
        return np.array([ids], dtype=np.int64)

    def _decode_ids(self, ids: list[int]) -> str:
        chars = []
        for idx in ids:
            if idx in (0, 1, 2, 3):
                continue
            sym = self.tgt_symbols[idx]
            chars.append(" " if sym == "_" else sym)
        return "".join(chars).strip()

    def _lookup_prob(self, word: str) -> float:
        row = self.db.execute("SELECT prob FROM probs WHERE word = ?", (word,)).fetchone()
        return float(row[0]) if row else 0.0

    def _expand_mixed_clusters(self, candidates: list[str], limit: int) -> list[str]:
        if not candidates:
            return candidates

        extra: list[str] = []
        seen = set(candidates)
        for cand in candidates[:2]:
            for src_cluster, dst_cluster in MIXED_CLUSTER_RULES:
                if src_cluster in cand:
                    variant = cand.replace(src_cluster, dst_cluster)
                    if variant not in seen:
                        seen.add(variant)
                        extra.append(variant)

        if not extra:
            return candidates[:limit]

        # Insert synthetic variant at slot #4 (index 3) so common #2/#3 candidates stay intact
        merged = list(candidates)
        insert_pos = min(3, len(merged))
        merged.insert(insert_pos, extra[0])
        return merged[:limit]

    @lru_cache(maxsize=4096)
    def _infer_cached(self, word: str, beam_width: int = 4) -> tuple[str, ...]:
        src_tokens = self._encode_word(word)
        enc_out = self.enc_sess.run(None, {"src_tokens": src_tokens})[0]
        enc_out_cache: dict[int, np.ndarray] = {1: enc_out}

        beams = [(0.0, [self.eos_idx], 0)]
        completed: list[tuple[float, list[int]]] = []

        past_k = np.zeros((self.num_layers, self.num_heads, 0, self.head_dim), dtype=np.float32)
        past_v = np.zeros((self.num_layers, self.num_heads, 0, self.head_dim), dtype=np.float32)

        max_len = len(word) + 6
        for step in range(max_len):
            if not beams:
                break

            bsz = len(beams)
            last_tok = np.array([[b[1][-1]] for b in beams], dtype=np.int64)
            step_idx = np.array([step], dtype=np.int64)
            enc_out_b = enc_out_cache.get(bsz)
            if enc_out_b is None:
                enc_out_b = np.repeat(enc_out, bsz, axis=1)
                enc_out_cache[bsz] = enc_out_b

            log_probs, pres_k, pres_v = self.dec_sess.run(
                None,
                {
                    "last_token": last_tok,
                    "step_idx": step_idx,
                    "encoder_out": enc_out_b,
                    "past_keys": past_k,
                    "past_values": past_v,
                },
            )

            next_candidates = []
            for i, (score, seq, _) in enumerate(beams):
                top_ids = np.argpartition(log_probs[i], -beam_width)[-beam_width:]
                for tid in top_ids:
                    tid_int = int(tid)
                    new_score = score + float(log_probs[i, tid_int])
                    new_seq = seq + [tid_int]
                    if tid_int == self.eos_idx:
                        completed.append((new_score / len(new_seq), new_seq))
                    else:
                        next_candidates.append((new_score, new_seq, i))

            next_candidates.sort(key=lambda x: x[0], reverse=True)

            if len(completed) >= beam_width:
                worst_done = sorted(completed, key=lambda x: x[0], reverse=True)[beam_width - 1][0]
                if not next_candidates or (next_candidates[0][0] / (step + 2)) < worst_done:
                    break

            beams = next_candidates[:beam_width]
            if not beams:
                break

            head_indices = []
            for _, _, p_idx in beams:
                head_indices.extend(range(p_idx * self.num_heads, (p_idx + 1) * self.num_heads))

            past_k = np.ascontiguousarray(pres_k[:, head_indices, :, :])
            past_v = np.ascontiguousarray(pres_v[:, head_indices, :, :])

        if not completed:
            completed = [(s / len(seq), seq) for s, seq, _ in beams]

        rescored = []
        seen = set()
        for model_score, seq in completed:
            text = self._decode_ids(seq)
            if not text or text in seen:
                continue
            seen.add(text)
            dict_bonus = self._lookup_prob(text)
            rescored.append((model_score + 0.35 * dict_bonus, text))

        rescored.sort(key=lambda x: x[0], reverse=True)
        base_results = [text for _, text in rescored[:beam_width]]
        expanded = self._expand_mixed_clusters(base_results, limit=5)
        return tuple(expanded) if expanded else (word,)

    def translit_word(self, word: str, beam_width: int = 4) -> tuple[str, ...]:
        key = word.lower()
        results = list(self._infer_cached(key, beam_width))

        extras = SHORT_SYLLABLE_EXTRAS.get(key)
        if extras:
            for extra in reversed(extras):
                if extra not in results:
                    results.insert(1, extra)

        preferred = self.user_prefs.get(key)
        if preferred:
            if preferred in results:
                results.remove(preferred)
            results.insert(0, preferred)

        exact = EXACT_CASE_OVERRIDES.get(word)
        if exact:
            if exact in results:
                results.remove(exact)
            results.insert(0, exact)

        return tuple(results[:5])

    def record_choice(self, word: str, chosen: str) -> None:
        key = word.lower()
        if not key or not chosen or chosen == word:
            return
        if word in EXACT_CASE_OVERRIDES and EXACT_CASE_OVERRIDES[word] == chosen:
            return
        if self.user_prefs.get(key) == chosen:
            return
        self.user_prefs[key] = chosen
        tmp_file = PREFS_FILE.with_suffix(".json.tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self.user_prefs, f, ensure_ascii=False, indent=2)
            tmp_file.replace(PREFS_FILE)
        except Exception:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    tr = KannadaOnnxTransliterator()
    print(f"Loaded in {(time.perf_counter() - t0) * 1000:.1f} ms")
    for w in ["doddu", "hoddu", "sandu", "na", "Na", "anna"]:
        t1 = time.perf_counter()
        print(f"{w:10s} -> {tr.translit_word(w)} ({(time.perf_counter() - t1) * 1000:.2f} ms)")
