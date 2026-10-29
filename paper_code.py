# ==============================================================================
# CORE: preprocessing, ciphers, features, models, evaluation, explanations
# Re-implementation of the pipeline described in the thesis (Chapters 3-4).
# Requirements: python>=3.9, numpy, pandas, scikit-learn, scipy, statsmodels,
#               tensorflow>=2.12, shap, lime, datasketch, matplotlib
# ==============================================================================
import os, re, json, time, pickle, random, string, warnings, argparse, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

CIPHERS = ["Caesar", "Playfair", "Rail Fence", "Vigenère"]          # class order (label = index)
BASELINES = ["Random Forest", "RF-Stat", "RF-Stat+", "CNN", "BiLSTM"]
PROPOSED = ["RF-CAF", "CAF-Net"]                                       # innovation (Section "Proposed method")
MODELS = BASELINES + PROPOSED
CATEGORIES = ["Short", "Medium", "Long"]                              # 10-50, 51-200, 201-500 chars
ALPHA = string.ascii_uppercase
CONDITIONS = {                                                         # name -> (format, key regime)
    "raw_fixed": ("raw", "fixed"), "raw_random": ("raw", "random"),
    "norm_fixed": ("norm", "fixed"), "norm_random": ("norm", "random")}
FIXED_KEYS = {"caesar": 3, "vigenere": "SECURITY", "playfair": "MONARCHY", "rail": 3}
ENGLISH_FREQ = np.array([8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966, 0.153, 0.772, 4.025,
                         2.406, 6.749, 7.507, 1.929, 0.095, 5.987, 6.327, 9.056, 2.758, 0.978, 2.360, 0.150,
                         1.974, 0.074]) / 100.0

DEFAULT_CFG = dict(
    seed_data=42, max_len=200, tfidf_features=100000, rf_trees_tfidf=50, rf_depth_tfidf=15,
    rf_trees_stat=200, epochs=15, patience=2, batch=128, val_split=0.1, shap_n_stat=300, shap_n_tfidf=100,
    lime_samples=1000, unseen_seed=777, bin_width=10, min_bin_n=20, threshold=0.95,
    # --- proposed method (CAF-Net) ---
    caf_epochs=20, caf_patience=3, branch_drop=0.3, conformal_alpha=0.05, cal_frac=0.1, feat_jobs=-1)

# ------------------------------------------------------------------------------
# 1. Dataset construction (Section 3.4.2, Table 3.2)
# ------------------------------------------------------------------------------
def extract_body(raw):
    parts = re.split(r"\r?\n\r?\n", str(raw), maxsplit=1)
    return parts[1] if len(parts) > 1 else ""

def clean_text(t):
    t = re.sub(r"(?is)-{2,}\s*(original message|forwarded by).*$", " ", t)   # forwarded / original blocks
    t = re.sub(r"(?m)^\s*>.*$", " ", t)                                       # quoted reply lines
    t = re.sub(r"<[^>]+>", " ", t)                                            # HTML tags
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)                              # URLs
    t = re.sub(r"\S+@\S+", " ", t)                                            # e-mail addresses
    t = t.encode("ascii", "ignore").decode()                                  # non-ASCII
    return re.sub(r"\s+", " ", t).strip()

def length_category(n):
    return "Short" if n <= 50 else ("Medium" if n <= 200 else "Long")

def minhash_dedup(texts, threshold=0.95, num_perm=128, shingle=5):
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = []
    for i, t in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for j in range(max(1, len(t) - shingle + 1)):
            m.update(t[j:j + shingle].encode("utf8"))
        if lsh.query(m):
            continue
        lsh.insert(str(i), m); keep.append(i)
    return keep

def build_dataset(emails_csv, out_dir, seed=42, nrows=None):
    """emails.csv (columns: file, message) -> balanced DataFrame [msg_id, text, plen, category]; writes Table 3.2."""
    os.makedirs(out_dir, exist_ok=True)
    raw = pd.read_csv(emails_csv, nrows=nrows)
    rows = [("Raw messages", len(raw))]
    bodies = raw["message"].map(extract_body).map(clean_text)
    bodies = bodies[bodies.str.len() > 0]; rows.append(("Non-empty cleaned bodies", len(bodies)))
    bodies = bodies.drop_duplicates(); rows.append(("After exact de-duplication", len(bodies)))
    bodies = bodies[bodies.str.len().between(10, 500)]; rows.append(("Length 10–500 characters", len(bodies)))
    bodies = bodies[bodies.map(lambda s: sum(c.isalpha() for c in s)) >= 6]; rows.append(("At least six letters", len(bodies)))
    keep = minhash_dedup(bodies.tolist()); bodies = bodies.iloc[keep]
    rows.append(("After near-duplicate removal (MinHash-LSH)", len(bodies)))
    df = pd.DataFrame({"text": bodies.values}); df["plen"] = df.text.str.len(); df["category"] = df.plen.map(length_category)
    n = df.category.value_counts().min()
    df = pd.concat([g.sample(n=n, random_state=seed) for _, g in df.groupby("category")]).reset_index(drop=True)
    df.insert(0, "msg_id", np.arange(len(df)))
    rows.append(("Balanced by length category", len(df)))
    rows.append(("Encrypted samples per condition (x 4 ciphers)", 4 * len(df)))
    pd.DataFrame(rows, columns=["Step", "Messages"]).to_csv(f"{out_dir}/Table_3_2_dataset_pipeline_counts.csv", index=False)
    return df

# ------------------------------------------------------------------------------
# 2. Ciphers (Section 3.4.3) - validated against textbook examples in self_test()
# ------------------------------------------------------------------------------
def normalise(text):
    return re.sub(r"[^A-Za-z]", "", text).upper()

def caesar(text, shift):
    out = []
    for c in text:
        if c.isalpha() and c.isascii():
            base = 65 if c.isupper() else 97
            out.append(chr((ord(c) - base + shift) % 26 + base))
        else: out.append(c)
    return "".join(out)

def vigenere(text, key):
    key = [ord(k) - 65 for k in key.upper()]; out = []; j = 0
    for c in text:
        if c.isalpha() and c.isascii():
            base = 65 if c.isupper() else 97
            out.append(chr((ord(c) - base + key[j % len(key)]) % 26 + base)); j += 1
        else: out.append(c)
    return "".join(out)

def playfair_square(keyword):
    seen = []
    for c in (keyword.upper().replace("J", "I") + ALPHA.replace("J", "")):
        if c.isalpha() and c not in seen: seen.append(c)
    return seen[:25]

def playfair(text, square):
    pos = {c: divmod(i, 5) for i, c in enumerate(square)}
    letters = [c for c in text.upper().replace("J", "I") if "A" <= c <= "Z"]
    pairs, i = [], 0
    while i < len(letters):
        a = letters[i]; b = letters[i + 1] if i + 1 < len(letters) else None
        if b is None: b = "X"; i += 1
        elif a == b: b = "X" if a != "X" else "Q"; i += 1
        else: i += 2
        pairs.append((a, b))
    out = []
    for a, b in pairs:
        (ra, ca), (rb, cb) = pos[a], pos[b]
        if ra == rb: out += [square[ra * 5 + (ca + 1) % 5], square[rb * 5 + (cb + 1) % 5]]
        elif ca == cb: out += [square[((ra + 1) % 5) * 5 + ca], square[((rb + 1) % 5) * 5 + cb]]
        else: out += [square[ra * 5 + cb], square[rb * 5 + ca]]
    return "".join(out)

def railfence(text, rails):
    if rails < 2 or rails >= len(text): return text
    rows = [[] for _ in range(rails)]; r, d = 0, 1
    for c in text:
        rows[r].append(c)
        if r == 0: d = 1
        elif r == rails - 1: d = -1
        r += d
    return "".join("".join(x) for x in rows)

def self_test():
    assert caesar("ATTACK", 3) == "DWWDFN"
    assert vigenere("ATTACKATDAWN", "LEMON") == "LXFOPVEFRNHR"
    assert playfair("HIDETHEGOLDINTHETREESTUMP", playfair_square("PLAYFAIREXAMPLE")) == "BMODZBXDNABEKUDMUIXMMOUVIF"
    assert railfence("WEAREDISCOVEREDFLEEATONCE", 3) == "WECRLTEERDSOEEFEAOCAIVDEN"
    return True

def random_keys(rng):
    return dict(caesar=int(rng.randint(1, 26)),
                vigenere="".join(rng.choice(list(ALPHA), size=int(rng.randint(3, 13)))),
                playfair="".join(rng.permutation(list(ALPHA.replace("J", "")))),
                rail=int(rng.randint(2, 9)))

def encrypt_dataset(df, fmt, key_regime, seed=42):
    """One row per (message, cipher). Returns DataFrame [msg_id, cipher, label, ciphertext, plen, category]."""
    rng = np.random.RandomState(seed); recs = []
    for r in df.itertuples():
        p = normalise(r.text) if fmt == "norm" else r.text
        k = FIXED_KEYS if key_regime == "fixed" else random_keys(rng)
        sq = playfair_square(k["playfair"])
        cts = {"Caesar": caesar(p, k["caesar"]), "Playfair": playfair(p, sq),
               "Rail Fence": railfence(p, k["rail"]), "Vigenère": vigenere(p, k["vigenere"])}
        for lab, c in enumerate(CIPHERS):
            recs.append((r.msg_id, c, lab, cts[c], r.plen, r.category))
    return pd.DataFrame(recs, columns=["msg_id", "cipher", "label", "ciphertext", "plen", "category"])

# ------------------------------------------------------------------------------
# 3. Features (Section 4.4)
# ------------------------------------------------------------------------------
def _ic(counts):
    n = counts.sum()
    return float((counts * (counts - 1)).sum() / (n * (n - 1))) if n > 1 else 0.0

def stat_features(s):
    up = s.upper(); let = [c for c in up if "A" <= c <= "Z"]; n = len(let)
    counts = np.bincount([ord(c) - 65 for c in let], minlength=26).astype(float) if n else np.zeros(26)
    p = counts / n if n else counts
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    exp = ENGLISH_FREQ * n
    chi = float(((counts - exp) ** 2 / np.maximum(exp, 1e-9)).sum() / n) if n else 0.0
    n_up = sum(1 for c in s if "A" <= c <= "Z")
    f = dict(IC=_ic(counts), Entropy=ent, Chi2_per_letter=chi, Space_ratio=s.count(" ") / max(len(s), 1),
             Upper_ratio=n_up / max(n, 1), NonLetter_ratio=(len(s) - n) / max(len(s), 1), Length=len(s))
    # --- additional indicators of RF-Stat+ (key- and format-independent) ---
    dig = [(let[i], let[i + 1]) for i in range(0, n - 1, 2)]
    f["Even_length"] = int(n % 2 == 0)
    f["Has_J"] = int("J" in let)
    f["Doubled_digraphs"] = (sum(1 for a, b in dig if a == b) / len(dig)) if dig else 0.0
    best = 0.0
    for per in range(2, 11):
        if n < per * 2: continue
        ics = [_ic(np.bincount([ord(c) - 65 for c in let[k::per]], minlength=26).astype(float)) for k in range(per)]
        best = max(best, float(np.mean(ics)))
    f["Max_periodic_IC"] = best
    if dig:
        _, cnt = np.unique(["".join(d) for d in dig], return_counts=True); m = cnt.sum()
        f["Digraphic_IC"] = float((cnt * (cnt - 1)).sum() / (m * (m - 1))) if m > 1 else 0.0
    else: f["Digraphic_IC"] = 0.0
    return f

STAT7 = ["IC", "Entropy", "Chi2_per_letter", "Space_ratio", "Upper_ratio", "NonLetter_ratio", "Length"]
STAT12 = STAT7 + ["Even_length", "Has_J", "Doubled_digraphs", "Max_periodic_IC", "Digraphic_IC"]

# ------------------------------------------------------------------------------
# 3b. INNOVATION (1): cryptanalytic attack-success features (key-invariant by construction)
#   Each cipher hypothesis is tested by a cheap key-search attack; the feature is how well the
#   best attack restores English structure. Because the key is searched, not learned, the features
#   do not depend on the key used for training (target: the unseen-key drop of CNN/BiLSTM).
#   English structure is scored with the 40 most frequent English bigrams (no external corpus).
# ------------------------------------------------------------------------------
TOP_BIGRAMS = ("TH HE IN ER AN RE ON AT EN ND TI ES OR TE OF ED IS IT AL AR ST TO NT NG SE HA AS OU IO LE "
               "VE CO ME DE HI RI RO IC NE EA").split()
_BIG = np.zeros(676, bool)
for _b in TOP_BIGRAMS: _BIG[(ord(_b[0]) - 65) * 26 + ord(_b[1]) - 65] = True
_ESHIFT = np.stack([np.roll(ENGLISH_FREQ, s) for s in range(26)])      # row s: expected cipher-letter freq under shift s

def _bigram_rate(L):
    return float(_BIG[L[:-1] * 26 + L[1:]].mean()) if len(L) > 1 else 0.0

def _shift_chi(C):
    """C: (p, 26) letter counts per column -> best shift per column and its chi2 per letter."""
    n = np.maximum(C.sum(1, keepdims=True), 1.0); E = n[:, None, :] * _ESHIFT[None]
    chi = ((C[:, None, :] - E) ** 2 / np.maximum(E, 1e-9)).sum(2)        # (p, 26)
    return chi.argmin(1), chi.min(1) / n[:, 0], chi

def rail_decrypt(ct, rails):
    n = len(ct)
    if rails < 2 or rails >= n: return ct
    cyc = 2 * (rails - 1); i = np.arange(n) % cyc; rail = np.where(i < rails, i, cyc - i)
    order = np.argsort(rail, kind="stable"); out = np.empty(n, dtype=object); out[order] = list(ct)
    return "".join(out)

def attack_features(s):
    up = s.upper(); L = np.array([ord(c) - 65 for c in up if "A" <= c <= "Z"], dtype=np.int64); n = len(L)
    f = dict(CA_bigram_raw=0.0, CA_caesar_chi=0.0, CA_caesar_gain=0.0, CA_caesar_bigram=0.0, CA_vig_bigram=0.0,
             CA_vig_chi=0.0, CA_vig_gain=0.0, CA_rail_bigram=0.0, CA_rail_gain=0.0, CA_pf_reverse_sym=0.0)
    if n < 2: return f
    f["CA_bigram_raw"] = _bigram_rate(L)
    # H1 Caesar: 26-shift search
    sh, chi_min, chi = _shift_chi(np.bincount(L, minlength=26).astype(float)[None])
    f["CA_caesar_chi"] = float(chi_min[0]); f["CA_caesar_gain"] = float((chi[0, 0] - chi[0].min()) / n)
    f["CA_caesar_bigram"] = _bigram_rate((L - sh[0]) % 26)
    # H2 Vigenere: period 2..12, per-column shift search, judged by bigrams (not by the unigram fit itself)
    best, best_chi = 0.0, 0.0
    for p in range(2, 13):
        if n < 2 * p: break
        C = np.stack([np.bincount(L[k::p], minlength=26) for k in range(p)]).astype(float)
        sh, cm, _ = _shift_chi(C); r = _bigram_rate((L - sh[np.arange(n) % p]) % 26)
        if r > best: best, best_chi = r, float(cm.mean())
    f["CA_vig_bigram"], f["CA_vig_chi"] = best, best_chi; f["CA_vig_gain"] = best - f["CA_caesar_bigram"]
    # H3 Rail Fence: undo 2..9 rails on the full string (the cipher transposes every character)
    best = 0.0
    for r in range(2, 10):
        if r >= len(s): break
        D = np.array([ord(c) - 65 for c in rail_decrypt(s, r).upper() if "A" <= c <= "Z"], dtype=np.int64)
        best = max(best, _bigram_rate(D))
    f["CA_rail_bigram"] = best; f["CA_rail_gain"] = best - f["CA_bigram_raw"]
    # H4 Playfair: reversed digraphs map to reversed digraphs (AB->XY implies BA->YX)
    dig = {(int(L[i]), int(L[i + 1])) for i in range(0, n - 1, 2) if L[i] != L[i + 1]}
    f["CA_pf_reverse_sym"] = (sum(1 for a, b in dig if (b, a) in dig) / len(dig)) if dig else 0.0
    return f

ATTACK = ["CA_bigram_raw", "CA_caesar_chi", "CA_caesar_gain", "CA_caesar_bigram", "CA_vig_bigram", "CA_vig_chi",
          "CA_vig_gain", "CA_rail_bigram", "CA_rail_gain", "CA_pf_reverse_sym"]
STAT_CAF = STAT12 + ATTACK

def _feat_chunk(texts):
    return [{**stat_features(t), **attack_features(t)} for t in texts]

def stat_matrix(texts, n_jobs=-1):
    if len(texts) < 5000 or n_jobs == 1: rows = _feat_chunk(texts)
    else:
        from joblib import Parallel, delayed
        chunks = [texts[i:i + 2000] for i in range(0, len(texts), 2000)]
        rows = [r for part in Parallel(n_jobs=n_jobs)(delayed(_feat_chunk)(c) for c in chunks) for r in part]
    return pd.DataFrame(rows)[STAT_CAF]

def build_vocab(texts):
    chars = sorted(set("".join(texts))); return {c: i + 1 for i, c in enumerate(chars)}

def encode_seq(texts, vocab, max_len):
    X = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        ids = [vocab.get(c, 0) for c in t[:max_len]]; X[i, :len(ids)] = ids
    return X

# ------------------------------------------------------------------------------
# 4. Models
# ------------------------------------------------------------------------------
def make_cnn(vocab_size, cfg):
    import tensorflow as tf
    L = tf.keras.layers
    m = tf.keras.Sequential([L.Input((cfg["max_len"],)), L.Embedding(vocab_size, 128), L.Conv1D(128, 5, activation="relu"),
                             L.GlobalMaxPooling1D(), L.Dense(64, activation="relu"), L.Dropout(0.5), L.Dense(4, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]); return m

def make_bilstm(vocab_size, cfg):
    import tensorflow as tf
    L = tf.keras.layers
    m = tf.keras.Sequential([L.Input((cfg["max_len"],)), L.Embedding(vocab_size, 64, mask_zero=True),
                             L.Bidirectional(L.LSTM(64)), L.Dense(64, activation="relu"), L.Dropout(0.3), L.Dense(4, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]); return m

def fit_neural(builder, Xtr, ytr, vocab_size, cfg, seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    m = builder(vocab_size, cfg)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"], restore_best_weights=True)]
    t0 = time.time()
    h = m.fit(Xtr, ytr, validation_split=cfg["val_split"], epochs=cfg["epochs"], batch_size=cfg["batch"], callbacks=cb, verbose=0)
    return m, time.time() - t0, {k: [float(x) for x in v] for k, v in h.history.items()}

# ------------------------------------------------------------------------------
# 4b. INNOVATION (2): CAF-Net - Cryptanalysis-Aware, length-gated Fusion Network
#   branch S: multi-scale character CNN (k = 3, 5, 7) on the ciphertext
#   branch C: MLP on the cryptanalytic indicators (STAT12 + attack-success features)
#   gate g = sigmoid(W [log-length, h_S, h_C]) ; fused = g * h_S + (1 - g) * h_C
#   branch dropout: during training h_S is zeroed for a random 30% of samples, so the
#   network cannot rely only on key-specific character patterns.
#   The gate is exported to show, per length, how much the decision uses each branch.
# ------------------------------------------------------------------------------
def make_caf(vocab_size, n_feat, cfg, use_gate=True, branch_drop=None, name="caf"):
    import tensorflow as tf
    L = tf.keras.layers; bd = cfg["branch_drop"] if branch_drop is None else branch_drop

    class BranchDrop(L.Layer):
        def __init__(self, rate, **kw): super().__init__(**kw); self.rate = rate
        def call(self, x, training=None):
            if not training or self.rate <= 0: return x
            keep = tf.cast(tf.random.uniform((tf.shape(x)[0], 1)) >= self.rate, x.dtype); return x * keep

    q = L.Input((cfg["max_len"],), name="seq"); s = L.Input((n_feat,), name="feat"); ln = L.Input((1,), name="loglen")
    e = L.Embedding(vocab_size, 64)(q)
    hs = L.Concatenate()([L.GlobalMaxPooling1D()(L.Conv1D(96, k, activation="relu", padding="same")(e)) for k in (3, 5, 7)])
    hs = BranchDrop(bd)(L.Dropout(0.3)(L.Dense(64, activation="relu")(hs)))
    hc = L.Dense(64, activation="relu")(L.Dense(128, activation="relu")(s))
    if use_gate:
        g = L.Dense(64, activation="sigmoid", name="gate")(L.Concatenate()([ln, hs, hc]))
        h = L.Add()([L.Multiply()([g, hs]), L.Multiply()([L.Lambda(lambda t: 1.0 - t)(g), hc])])
    else:
        g = None; h = L.Concatenate()([hs, hc, ln])                    # length still used, only the gate is removed
    out = L.Dense(4, activation="softmax")(L.Dropout(0.3)(L.Dense(64, activation="relu")(h)))
    m = tf.keras.Model([q, s, ln], out, name=name)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    gm = tf.keras.Model([q, s, ln], L.Lambda(lambda t: tf.reduce_mean(t, axis=1))(g)) if use_gate else None
    return m, gm

def caf_inputs(Q, F, lengths):
    return [Q, F.astype(np.float32), (np.log1p(np.asarray(lengths, float)) / np.log(501.0)).reshape(-1, 1).astype(np.float32)]

def fit_caf(Xfit, yfit, Xval, yval, vocab_size, n_feat, cfg, seed, **kw):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed); m, gm = make_caf(vocab_size, n_feat, cfg, **kw)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["caf_patience"], restore_best_weights=True)]
    t0 = time.time()
    h = m.fit(Xfit, yfit, validation_data=(Xval, yval), epochs=cfg["caf_epochs"], batch_size=cfg["batch"], callbacks=cb, verbose=0)
    return m, gm, time.time() - t0, {k: [float(x) for x in v] for k, v in h.history.items()}

# ------------------------------------------------------------------------------
# 4c. INNOVATION (3): length-conditional (Mondrian) conformal prediction
#   Instead of one "minimum reliable length", every prediction gets a set of ciphers that
#   contains the true cipher with probability >= 1 - alpha *within each length category*.
#   Long texts -> singleton sets; short texts -> larger sets (honest uncertainty).
# ------------------------------------------------------------------------------
def conformal_fit(p_cal, y_cal, groups, alpha):
    s = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]; q = {}
    for g in np.unique(groups):
        sg = np.sort(s[groups == g]); n = len(sg); k = int(np.ceil((n + 1) * (1 - alpha))) - 1
        q[g] = 1.0 if k >= n else float(sg[k])
    k = int(np.ceil((len(s) + 1) * (1 - alpha))) - 1; q["__all__"] = 1.0 if k >= len(s) else float(np.sort(s)[k])
    return q

def conformal_eval(p, y, groups, q, tag, cond, seed):
    rows = []
    for mode in ("Mondrian (by length)", "Marginal"):
        thr = np.array([q.get(g, q["__all__"]) if mode.startswith("Mondrian") else q["__all__"] for g in groups])
        S = (1.0 - p) <= thr[:, None]; S[np.arange(len(y)), p.argmax(1)] = True; cov = S[np.arange(len(y)), y]; size = S.sum(1)
        for cat in CATEGORIES + ["All"]:
            m = np.ones(len(y), bool) if cat == "All" else (groups == cat)
            if m.sum() == 0: continue
            rows.append(dict(Condition=cond, Seed=seed, Test=tag, Mode=mode, Category=cat, Coverage=100 * cov[m].mean(),
                             Avg_set_size=size[m].mean(), Singleton_rate=100 * (size[m] == 1).mean(),
                             Singleton_accuracy=100 * cov[m & (size == 1)].mean() if (m & (size == 1)).any() else float("nan")))
    return rows

# ------------------------------------------------------------------------------
# 5. Metrics, length analysis, statistics
# ------------------------------------------------------------------------------
def metrics(y, pred, proba):
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    from statsmodels.stats.proportion import proportion_confint
    n = len(y); ok = int((y == pred).sum())
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    try: auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted") * 100
    except Exception: auc = float("nan")
    lo, hi = proportion_confint(ok, n, method="wilson")
    return dict(N_test=n, Accuracy=100 * ok / n, Precision=100 * p, Recall=100 * r, F1=100 * f, ROC_AUC=auc,
                Errors=n - ok, CI_low=100 * lo, CI_high=100 * hi)

def min_reliable_length(plen, correct, thr=0.95, width=10, min_n=20):
    b = (np.asarray(plen) // width) * width; df = pd.DataFrame({"b": b, "ok": np.asarray(correct, float)})
    g = df.groupby("b").ok.agg(["mean", "count"]); g = g[g["count"] >= min_n].sort_index()
    if g.empty or g["mean"].iloc[-1] < thr: return "not reached"
    start = g.index[-1]
    for bi in g.index[::-1]:
        if g.loc[bi, "mean"] >= thr: start = bi
        else: break
    return int(start)

def mcnemar_table(preds, y):
    from scipy.stats import chi2
    rows = []
    for a, b in itertools.combinations([m for m in MODELS if m in preds], 2):
        ca, cb = preds[a] == y, preds[b] == y
        n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01) if n10 + n01 > 0 else 0.0
        rows.append(dict(Pair=f"{a} vs {b}", First_right_second_wrong=n10, Second_right_first_wrong=n01,
                         Chi2=stat, p_value=float(chi2.sf(stat, 1))))
    return pd.DataFrame(rows)


def fig_acc_vs_length(cdir, cond, Ete, preds, y_te, cfg):
    """Figure 5.5: per-cipher accuracy as a function of plaintext length (10-character bins; dashed line = 95%)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ms = [m for m in MODELS if m in preds]
    fig, axs = plt.subplots(1, len(ms), figsize=(4 * len(ms), 3.4), sharey=True); b = (Ete.plen.values // cfg["bin_width"]) * cfg["bin_width"]
    for ax, m in zip(axs, ms):
        for k, c in enumerate(CIPHERS):
            d = pd.DataFrame({"b": b[y_te == k], "ok": (preds[m] == y_te)[y_te == k] * 100.0}).groupby("b").ok.agg(["mean", "count"])
            d = d[d["count"] >= cfg["min_bin_n"]]; ax.plot(d.index, d["mean"], label=c)
        ax.axhline(95, color="gray", ls="--", lw=1); ax.set_title(m); ax.set_xlabel("Plaintext length (chars)")
    axs[0].set_ylabel("Accuracy (%)"); axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{cdir}/Figure_5_5_accuracy_vs_length.png", dpi=300); plt.close(fig)

# ------------------------------------------------------------------------------
# 6. One experimental condition x one seed
# ------------------------------------------------------------------------------
def feature_cache(enc, tag, out_dir):
    f = f"{out_dir}/cache_feat_v2_{tag}.pkl"                           # v2 = STAT12 + attack features
    if os.path.exists(f): return pickle.load(open(f, "rb"))
    t0 = time.time(); X = stat_matrix(enc.ciphertext.tolist(), n_jobs=DEFAULT_CFG["feat_jobs"])
    print(f"[features] {tag}: {len(X)} texts in {time.time() - t0:.0f}s"); pickle.dump(X, open(f, "wb")); return X

def run_seed(cond, seed, df, enc, stat_all, unseen_enc, unseen_stat, cfg, out_dir, first_seed):
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    run_file = f"{out_dir}/runs_v2/{cond}_{seed}.pkl"
    if os.path.exists(run_file): print(f"[skip] {cond} seed {seed} (checkpoint)"); return pickle.load(open(run_file, "rb"))
    cdir = f"{out_dir}/{cond}"; os.makedirs(cdir, exist_ok=True); os.makedirs(f"{out_dir}/runs_v2", exist_ok=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(enc, enc.label, groups=enc.msg_id))
    Etr, Ete = enc.iloc[tr], enc.iloc[te]; y_tr, y_te = Etr.label.values, Ete.label.values
    test_msgs = set(Ete.msg_id)
    R = dict(overall=[], length=[], cipher=[], minlen=[], unseen=[], epochs=[], conformal=[], gate=[], ablation=[])
    preds, probas, models = {}, {}, {}
    def record(name, pred, proba, secs):
        d = dict(Condition=cond, Seed=seed, Model=name, Train_s=secs); d.update(metrics(y_te, pred, proba)); R["overall"].append(d)
        preds[name], probas[name] = pred, proba
        ok = pred == y_te
        for cat in CATEGORIES:
            m = (Ete.category == cat).values; R["length"].append(dict(Condition=cond, Seed=seed, Model=name, Category=cat, Accuracy=100 * ok[m].mean()))
            for k, c in enumerate(CIPHERS):
                mm = m & (y_te == k); R["cipher"].append(dict(Condition=cond, Seed=seed, Model=name, Category=cat, Cipher=c, Recall=100 * ok[mm].mean()))
        ml = dict(Condition=cond, Seed=seed, Model=name)
        for k, c in enumerate(CIPHERS):
            mm = y_te == k; ml[c] = min_reliable_length(Ete.plen.values[mm], ok[mm], cfg["threshold"], cfg["bin_width"], cfg["min_bin_n"])
        R["minlen"].append(ml)
        if first_seed:
            pd.DataFrame(confusion_matrix(y_te, pred, labels=range(4)), index=CIPHERS, columns=CIPHERS).to_csv(f"{cdir}/CM_{name}.csv")
            p_, r_, f_, _ = precision_recall_fscore_support(y_te, pred, labels=range(4), zero_division=0)
            pd.DataFrame({"Cipher": CIPHERS, "Precision": (100 * p_).round(2), "Recall": (100 * r_).round(2), "F1": (100 * f_).round(2)}
                         ).to_csv(f"{cdir}/Table_5_2_per_cipher_{name}.csv", index=False)
    def unseen_eval(name, predict_fn):
        sel = unseen_enc.msg_id.isin(test_msgs).values; U = unseen_enc[sel]; yu = U.label.values
        pu = predict_fn(sel); ok = pu == yu
        d = dict(Condition=cond, Seed=seed, Model=name, Acc_seen=[o for o in R["overall"] if o["Model"] == name][0]["Accuracy"], Acc_unseen=100 * ok.mean())
        for k, c in enumerate(CIPHERS): d[f"Recall_{c}"] = 100 * ok[yu == k].mean()
        R["unseen"].append(d)
        if first_seed:
            pd.DataFrame(confusion_matrix(yu, pu, labels=range(4)), index=CIPHERS, columns=CIPHERS).to_csv(f"{cdir}/CM_unseen_keys_{name}.csv")
    do_unseen = unseen_enc is not None
    # ---- RF on TF-IDF n-grams
    t0 = time.time()
    tf = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=cfg["tfidf_features"], lowercase=False, dtype=np.float32)
    Xtr = tf.fit_transform(Etr.ciphertext); Xte = tf.transform(Ete.ciphertext)
    rf = RandomForestClassifier(n_estimators=cfg["rf_trees_tfidf"], max_depth=cfg["rf_depth_tfidf"], max_features="sqrt", random_state=seed, n_jobs=-1).fit(Xtr, y_tr)
    secs = time.time() - t0; record("Random Forest", rf.predict(Xte), rf.predict_proba(Xte), secs); models["rf"] = (rf, tf)
    if do_unseen: unseen_eval("Random Forest", lambda sel: rf.predict(tf.transform(unseen_enc.ciphertext[sel])))
    # ---- RF-Stat and RF-Stat+
    S_tr, S_te = stat_all.iloc[tr], stat_all.iloc[te]
    for name, cols in (("RF-Stat", STAT7), ("RF-Stat+", STAT12)):
        t0 = time.time(); m = RandomForestClassifier(n_estimators=cfg["rf_trees_stat"], random_state=seed, n_jobs=-1).fit(S_tr[cols], y_tr)
        record(name, m.predict(S_te[cols]), m.predict_proba(S_te[cols]), time.time() - t0); models[name] = m
        if do_unseen: unseen_eval(name, lambda sel, m=m, cols=cols: m.predict(unseen_stat[cols][sel]))
    # ---- CNN and BiLSTM
    vocab = build_vocab(Etr.ciphertext.tolist()); vs = len(vocab) + 1
    Q_tr = encode_seq(Etr.ciphertext.tolist(), vocab, cfg["max_len"]); Q_te = encode_seq(Ete.ciphertext.tolist(), vocab, cfg["max_len"])
    hist = {}
    for name, builder in (("CNN", make_cnn), ("BiLSTM", make_bilstm)):
        m, secs, h = fit_neural(builder, Q_tr, y_tr, vs, cfg, seed); pr = m.predict(Q_te, batch_size=512, verbose=0)
        record(name, pr.argmax(1), pr, secs); R["epochs"].append(dict(Condition=cond, Seed=seed, Model=name, Epochs=len(h["loss"]))); hist[name] = h; models[name] = m
        if do_unseen:
            unseen_eval(name, lambda sel, m=m: m.predict(encode_seq(unseen_enc.ciphertext[sel].tolist(), vocab, cfg["max_len"]), batch_size=512, verbose=0).argmax(1))
    # ==== PROPOSED (1): RF-CAF = RF on STAT12 + cryptanalytic attack-success features
    t0 = time.time(); m = RandomForestClassifier(n_estimators=cfg["rf_trees_stat"], random_state=seed, n_jobs=-1).fit(S_tr[STAT_CAF], y_tr)
    record("RF-CAF", m.predict(S_te[STAT_CAF]), m.predict_proba(S_te[STAT_CAF]), time.time() - t0); models["RF-CAF"] = m
    if do_unseen: unseen_eval("RF-CAF", lambda sel, m=m: m.predict(unseen_stat[STAT_CAF][sel]))
    # ==== PROPOSED (2): CAF-Net (+ conformal calibration on a held-out, group-aware part of the training split)
    from sklearn.preprocessing import StandardScaler
    fi, rest = next(GroupShuffleSplit(n_splits=1, test_size=2 * cfg["cal_frac"], random_state=seed).split(Etr, y_tr, groups=Etr.msg_id))
    vi, ci = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(rest, groups=Etr.msg_id.values[rest])); vi, ci = rest[vi], rest[ci]
    clen_tr, clen_te = Etr.ciphertext.str.len().values, Ete.ciphertext.str.len().values      # observable length (no plaintext needed)
    g_cal = np.array([length_category(n) for n in clen_tr[ci]]); g_te = np.array([length_category(n) for n in clen_te])
    sel_u = unseen_enc.msg_id.isin(test_msgs).values if do_unseen else None
    U = unseen_enc[sel_u] if do_unseen else None
    Q_u = encode_seq(U.ciphertext.tolist(), vocab, cfg["max_len"]) if do_unseen else None
    sub = lambda X, idx: [x[idx] for x in X]
    def caf_X(cols):
        sc = StandardScaler().fit(S_tr[cols].values[fi])
        Xa = caf_inputs(Q_tr, sc.transform(S_tr[cols].values), clen_tr); Xt = caf_inputs(Q_te, sc.transform(S_te[cols].values), clen_te)
        Xu = caf_inputs(Q_u, sc.transform(unseen_stat[cols].values[sel_u]), U.ciphertext.str.len().values) if do_unseen else None
        return Xa, Xt, Xu
    Xa, Xt, Xu = caf_X(STAT_CAF)
    m, gm, secs, h = fit_caf(sub(Xa, fi), y_tr[fi], sub(Xa, vi), y_tr[vi], vs, len(STAT_CAF), cfg, seed)
    pr = m.predict(Xt, batch_size=512, verbose=0); record("CAF-Net", pr.argmax(1), pr, secs)
    R["epochs"].append(dict(Condition=cond, Seed=seed, Model="CAF-Net", Epochs=len(h["loss"]))); hist["CAF-Net"] = h; models["CAF-Net"] = m
    q = conformal_fit(m.predict(sub(Xa, ci), batch_size=512, verbose=0), y_tr[ci], g_cal, cfg["conformal_alpha"])
    R["conformal"] += conformal_eval(pr, y_te, g_te, q, "Seen keys", cond, seed)
    gv = gm.predict(Xt, batch_size=512, verbose=0).ravel(); gb = (clen_te // cfg["bin_width"]) * cfg["bin_width"]
    for (k, b_), v in pd.DataFrame({"k": y_te, "b": gb, "g": gv}).groupby(["k", "b"]).g.agg(["mean", "count"]).iterrows():
        R["gate"].append(dict(Condition=cond, Seed=seed, Cipher=CIPHERS[int(k)], Length_bin=int(b_), Gate_char_branch=float(v["mean"]), N=int(v["count"])))
    pu = None
    if do_unseen:
        pu = m.predict(Xu, batch_size=512, verbose=0); unseen_eval("CAF-Net", lambda sel: pu.argmax(1))
        R["conformal"] += conformal_eval(pu, U.label.values, np.array([length_category(n) for n in U.ciphertext.str.len().values]), q,
                                         "Unseen keys (distribution shift)", cond, seed)
    # ==== Ablation of CAF-Net (optional: --ablation)
    if cfg.get("ablation"):
        def ab_row(name, p_te, p_u):
            ok = p_te.argmax(1) == y_te; cat = Ete.category.values
            d = dict(Condition=cond, Seed=seed, Variant=name, Accuracy=100 * ok.mean(), Acc_Short=100 * ok[cat == "Short"].mean(), Acc_Long=100 * ok[cat == "Long"].mean())
            if p_u is not None: d["Acc_unseen"] = 100 * (p_u.argmax(1) == U.label.values).mean()
            R["ablation"].append(d)
        ab_row("CAF-Net (full)", pr, pu)
        for name, cols, kw in (("w/o attack features (STAT12 only)", STAT12, {}), ("w/o length gate (concatenation)", STAT_CAF, dict(use_gate=False)),
                               ("w/o branch dropout", STAT_CAF, dict(branch_drop=0.0))):
            A, T_, Uu = (Xa, Xt, Xu) if cols is STAT_CAF else caf_X(cols)
            ma, _, _, _ = fit_caf(sub(A, fi), y_tr[fi], sub(A, vi), y_tr[vi], vs, len(cols), cfg, seed, **kw)
            ab_row(name, ma.predict(T_, batch_size=512, verbose=0), ma.predict(Uu, batch_size=512, verbose=0) if do_unseen else None)
    if first_seed:
        json.dump(hist, open(f"{cdir}/training_history.json", "w"))
        mcnemar_table(preds, y_te).to_csv(f"{cdir}/Table_5_8_McNemar.csv", index=False)
        explain(cond, cdir, cfg, models, vocab, Ete, Xte, S_te, y_te, seed, df.set_index("msg_id").text)
        fig_acc_vs_length(cdir, cond, Ete, preds, y_te, cfg)
    pickle.dump(R, open(run_file, "wb"))
    return R

# ------------------------------------------------------------------------------
# 7. Explanations: TreeSHAP (RF-Stat+, TF-IDF RF) and LIME (CNN)  (Section 4.4.5)
# ------------------------------------------------------------------------------
def _shap_values(expl, X):
    sv = expl.shap_values(X, approximate=True)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
    return sv                                                             # (n, features, classes)

def explain(cond, cdir, cfg, models, vocab, Ete, Xte, S_te, y_te, seed, plain=None):
    import shap
    from lime.lime_text import LimeTextExplainer
    rng = np.random.RandomState(seed)
    # RF-Stat+ : mean |SHAP| per indicator and class
    idx = rng.choice(len(S_te), size=min(cfg["shap_n_stat"], len(S_te)), replace=False)
    sv = _shap_values(shap.TreeExplainer(models["RF-Stat+"]), S_te[STAT12].iloc[idx])
    pd.DataFrame(np.abs(sv).mean(0), index=STAT12, columns=CIPHERS).round(4).to_csv(f"{cdir}/SHAP_RF_Stat_plus.csv")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        pd.read_csv(f"{cdir}/SHAP_RF_Stat_plus.csv", index_col=0).plot.barh(figsize=(8, 5), title=f"RF-Stat+ — {cond}")
        plt.xlabel("mean |SHAP|"); plt.tight_layout(); plt.savefig(f"{cdir}/Figure_5_6_SHAP_RF_Stat_plus.png", dpi=300); plt.close()
    except Exception as e: print("SHAP figure skipped:", e)
    # RF-CAF (proposed): which attack-success features drive each cipher decision
    if "RF-CAF" in models:
        sv = _shap_values(shap.TreeExplainer(models["RF-CAF"]), S_te[STAT_CAF].iloc[idx])
        pd.DataFrame(np.abs(sv).mean(0), index=STAT_CAF, columns=CIPHERS).round(4).to_csv(f"{cdir}/SHAP_RF_CAF.csv")
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            pd.read_csv(f"{cdir}/SHAP_RF_CAF.csv", index_col=0).plot.barh(figsize=(8, 7), title=f"RF-CAF — {cond}")
            plt.xlabel("mean |SHAP|"); plt.tight_layout(); plt.savefig(f"{cdir}/Figure_SHAP_RF_CAF.png", dpi=300); plt.close()
        except Exception as e: print("SHAP figure skipped:", e)
    # TF-IDF RF : top n-grams per class (chunks keep memory low)
    rf, tf = models["rf"]; names = np.array(tf.get_feature_names_out())
    idx = rng.choice(Xte.shape[0], size=min(cfg["shap_n_tfidf"], Xte.shape[0]), replace=False)
    expl = shap.TreeExplainer(rf); acc = None
    for ch in np.array_split(idx, max(1, len(idx) // 20)):
        s = np.abs(_shap_values(expl, Xte[ch].toarray())).sum(0); acc = s if acc is None else acc + s
    acc /= len(idx); rows = []
    for k, c in enumerate(CIPHERS):
        for j in np.argsort(-acc[:, k])[:10]: rows.append((c, repr(names[j]), float(acc[j, k])))
    pd.DataFrame(rows, columns=["Cipher", "Ngram", "Mean_abs_SHAP"]).to_csv(f"{cdir}/SHAP_tfidf_top_ngrams.csv", index=False)
    # LIME on the CNN : one short test message, four encryptions
    cnn = models["CNN"]
    def predict_fn(texts): return cnn.predict(encode_seq(list(texts), vocab, cfg["max_len"]), batch_size=256, verbose=0)
    short = Ete[(Ete.category == "Short")]; lime_rows = []
    def lime_group(grp):
        ex = LimeTextExplainer(class_names=CIPHERS, char_level=True, random_state=seed)
        for r in grp.itertuples():
            e = ex.explain_instance(r.ciphertext, predict_fn, num_features=10, num_samples=cfg["lime_samples"], labels=(r.label,))
            for ch, w in e.as_list(label=r.label):
                lime_rows.append((r.cipher, r.ciphertext, plain.loc[r.msg_id] if plain is not None else "", ch, w))
    fallback = None
    for mid in short.msg_id.unique():                      # first short message correctly classified under all four ciphers
        grp = short[short.msg_id == mid]
        if len(grp) < 4: continue
        if fallback is None: fallback = grp
        if all(predict_fn([r.ciphertext]).argmax(1)[0] == r.label for r in grp.itertuples()):
            lime_group(grp); break
    if not lime_rows and fallback is not None: lime_group(fallback)   # fallback: first short message
    pd.DataFrame(lime_rows, columns=["Cipher", "Ciphertext", "Plaintext", "Char", "Weight"]).to_csv(f"{cdir}/LIME_cnn.csv", index=False)

# ------------------------------------------------------------------------------
# 8. Driver for one condition (all seeds)
# ------------------------------------------------------------------------------
def run_condition(cond, seeds, df, cfg, out_dir):
    fmt, keys = CONDITIONS[cond]
    os.makedirs(f"{out_dir}/{cond}", exist_ok=True); os.makedirs(f"{out_dir}/runs_v2", exist_ok=True)
    enc = encrypt_dataset(df, fmt, keys, seed=cfg["seed_data"])
    stat_all = feature_cache(enc, cond, out_dir)
    # Table 5.3: mean statistical indicators by cipher and length category
    t53 = stat_all.copy(); t53["cipher"], t53["category"] = enc.cipher.values, enc.category.values
    t53.groupby(["cipher", "category"])[["IC", "Entropy", "Chi2_per_letter", "Space_ratio", "Upper_ratio"]].mean().round(4).reset_index() \
        .rename(columns={"cipher": "cipher", "category": "category"}).to_csv(f"{out_dir}/{cond}/Table_5_3_statistical_indicators.csv", index=False)
    unseen_enc = unseen_stat = None
    if keys == "fixed":                                  # unseen-key test set: same messages, random keys (RQ4)
        unseen_enc = encrypt_dataset(df, fmt, "random", seed=cfg["unseen_seed"]); unseen_stat = feature_cache(unseen_enc, cond + "_unseen", out_dir)
    for i, s in enumerate(seeds):
        t0 = time.time(); run_seed(cond, s, df, enc, stat_all, unseen_enc, unseen_stat, cfg, out_dir, first_seed=(i == 0))
        print(f"[done] {cond} seed {s} in {time.time() - t0:.0f}s")

def load_runs(out_dir):
    D = {k: [] for k in ["overall", "length", "cipher", "minlen", "unseen", "epochs", "conformal", "gate", "ablation"]}
    for f in sorted(os.listdir(f"{out_dir}/runs_v2")):
        if f.endswith(".pkl"):
            r = pickle.load(open(f"{out_dir}/runs_v2/{f}", "rb"))
            for k in D: D[k] += r.get(k, [])
    return {k: pd.DataFrame(v) for k, v in D.items()}

# ------------------------------------------------------------------------------
# 9. Aggregated tables and figures (Chapter 5 of the thesis / results of the paper)
# ------------------------------------------------------------------------------
def _plt():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}); return plt

COL = {"Random Forest": "#7f7f7f", "RF-Stat": "#9ecae1", "RF-Stat+": "#3182bd", "CNN": "#e6550d", "BiLSTM": "#31a354",
       "RF-CAF": "#756bb1", "CAF-Net": "#d62728"}

def _mean_sd(s):
    return f"{s.mean():.2f}" if len(s) < 2 else f"{s.mean():.2f} ± {s.std(ddof=1):.2f}"

def aggregate_tables(D, out_dir, conds):
    os.makedirs(f"{out_dir}/tables", exist_ok=True); T = f"{out_dir}/tables"
    ov = D["overall"]; ov = ov[ov.Condition.isin(conds)]
    rows = []
    for c in conds:
        for m in MODELS:
            s = ov[(ov.Condition == c) & (ov.Model == m)]
            if s.empty: continue
            rows.append(dict(Condition=c, Model=m, Seeds=len(s), Accuracy=_mean_sd(s.Accuracy), Precision=f"{s.Precision.mean():.2f}", F1=_mean_sd(s.F1),
                             ROC_AUC=f"{s.ROC_AUC.mean():.2f}", Errors=int(round(s.Errors.mean())), Train_s=f"{s.Train_s.mean():.0f}"))
    pd.DataFrame(rows).to_csv(f"{T}/Table_overall_accuracy.csv", index=False)
    ln = D["length"]; rows = []
    for c in conds:
        for m in MODELS:
            r = dict(Condition=c, Model=m)
            for cat in CATEGORIES:
                s = ln[(ln.Condition == c) & (ln.Model == m) & (ln.Category == cat)].Accuracy
                if not s.empty: r[cat] = _mean_sd(s)
            rows.append(r)
    pd.DataFrame(rows).to_csv(f"{T}/Table_accuracy_by_length.csv", index=False)
    ci = D["cipher"]
    ci[ci.Condition.isin(conds)].groupby(["Condition", "Model", "Cipher", "Category"]).Recall.mean().round(2).unstack("Category")[CATEGORIES] \
        .to_csv(f"{T}/Table_recall_by_cipher_and_length.csv")
    ml = D["minlen"]; rows = []
    for c in conds:
        for cph in CIPHERS:
            r = dict(Condition=c, Cipher=cph)
            for m in MODELS:
                v = ml[(ml.Condition == c) & (ml.Model == m)][cph].astype(str).tolist()
                key = lambda x: 10 ** 9 if x == "not reached" else int(float(x)); v = sorted(v, key=key)
                lab = lambda x: "n.r." if x == "not reached" else str(int(float(x)))
                r[m] = lab(v[0]) if len(set(v)) == 1 else f"{lab(v[0])}–{lab(v[-1])}"
            rows.append(r)
    pd.DataFrame(rows).to_csv(f"{T}/Table_min_reliable_length.csv", index=False)
    un = D["unseen"]
    if not un.empty:
        un[un.Condition.isin(conds)].groupby(["Condition", "Model"]).mean(numeric_only=True).drop(columns="Seed").round(2).to_csv(f"{T}/Table_unseen_keys.csv")
    ov.pivot_table(index=["Condition", "Model"], columns="Seed", values="Accuracy").round(2).to_csv(f"{T}/Table_accuracy_per_seed.csv")
    # --- proposed method: conformal reliability, ablation, gate behaviour
    cf = D["conformal"]
    if not cf.empty:
        cf = cf[cf.Condition.isin(conds)]
        cf.groupby(["Condition", "Test", "Mode", "Category"], sort=False)[["Coverage", "Avg_set_size", "Singleton_rate", "Singleton_accuracy"]] \
          .mean().round(2).to_csv(f"{T}/Table_conformal_CAF_Net.csv")
    ab = D["ablation"]
    if not ab.empty:
        ab[ab.Condition.isin(conds)].groupby(["Condition", "Variant"], sort=False).mean(numeric_only=True).drop(columns="Seed").round(2) \
          .to_csv(f"{T}/Table_ablation_CAF_Net.csv")
    gt = D["gate"]
    if not gt.empty:
        gt = gt[gt.Condition.isin(conds)].copy(); gt["Category"] = gt.Length_bin.map(lambda b: length_category(b + 1))
        gt.groupby(["Condition", "Cipher", "Category"]).apply(lambda d: np.average(d.Gate_char_branch, weights=d.N)).round(3) \
          .unstack("Category").reindex(columns=CATEGORIES).to_csv(f"{T}/Table_gate_char_branch.csv")
    return T

def fig_gate(D, out_dir, conds):
    gt = D["gate"]
    if gt.empty: return
    plt = _plt(); cs = [c for c in conds if c in gt.Condition.values]
    fig, axs = plt.subplots(1, len(cs), figsize=(4.4 * len(cs), 3.2), sharey=True, squeeze=False)
    for ax, c in zip(axs[0], cs):
        g = gt[(gt.Condition == c) & (gt.N >= 20)]
        for k in CIPHERS:
            d = g[g.Cipher == k].groupby("Length_bin").Gate_char_branch.mean(); ax.plot(d.index, d.values, label=k)
        ax.axhline(0.5, color="gray", ls=":", lw=1); ax.set_title(f"CAF-Net gate - {c}"); ax.set_xlabel("Ciphertext length (chars)")
    axs[0][0].set_ylabel("Weight on character branch (g)"); axs[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f"{out_dir}/Figure_CAF_gate_vs_length.png", dpi=300); plt.close(fig)

def fig_conformal(D, out_dir, conds):
    cf = D["conformal"]
    if cf.empty: return
    plt = _plt(); cf = cf[(cf.Mode.str.startswith("Mondrian")) & (cf.Category != "All") & (cf.Condition.isin(conds))]
    tests = list(dict.fromkeys(cf.Test)); fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for t, ls in zip(tests, ["-", "--"]):
        g = cf[cf.Test == t].groupby(["Condition", "Category"])[["Coverage", "Avg_set_size"]].mean()
        for c in [c for c in conds if c in g.index.get_level_values(0)]:
            v = g.loc[c].reindex(CATEGORIES); axs[0].plot(CATEGORIES, v.Coverage, ls, marker="o", label=f"{c} / {t}"); axs[1].plot(CATEGORIES, v.Avg_set_size, ls, marker="o")
    axs[0].axhline(95, color="gray", ls=":", lw=1); axs[0].set_ylabel("Coverage (%)"); axs[1].set_ylabel("Average set size (ciphers)")
    axs[0].legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig(f"{out_dir}/Figure_CAF_conformal.png", dpi=300); plt.close(fig)

def fig_overall(D, out_dir, conds):
    plt = _plt(); ov = D["overall"]; fig, ax = plt.subplots(figsize=(9.5, 3.8)); w = 0.8 / len(MODELS)
    for j, m in enumerate(MODELS):
        xs, ys, es = [], [], []
        for i, c in enumerate(conds):
            s = ov[(ov.Condition == c) & (ov.Model == m)].Accuracy
            if s.empty: continue
            xs.append(i + (j - (len(MODELS) - 1) / 2) * w); ys.append(s.mean()); es.append(0 if len(s) < 2 else s.std(ddof=1))
        ax.bar(xs, ys, w, yerr=es, label=m, color=COL[m], capsize=2)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds); ax.set_ylim(75, 101); ax.set_ylabel("Accuracy (%)"); ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{out_dir}/Figure_overall_accuracy.png", dpi=300); plt.close(fig)

def fig_confusion(out_dir, cond):
    plt = _plt(); fig, axs = plt.subplots(1, len(MODELS), figsize=(2.6 * len(MODELS), 2.9))
    for ax, m in zip(axs, MODELS):
        f = f"{out_dir}/{cond}/CM_{m}.csv"
        if not os.path.exists(f): continue
        c = pd.read_csv(f, index_col=0); p = c.div(c.sum(axis=1), axis=0) * 100
        ax.imshow(p.values, cmap="Blues", vmin=0, vmax=100)
        for i in range(4):
            for j in range(4): ax.text(j, i, f"{p.values[i, j]:.1f}\n({int(c.values[i, j]):,})", ha="center", va="center", fontsize=6, color="white" if p.values[i, j] > 55 else "black")
        ax.set_xticks(range(4)); ax.set_xticklabels(["Cae", "Play", "Rail", "Vig"]); ax.set_yticks(range(4)); ax.set_yticklabels(["Cae", "Play", "Rail", "Vig"]); ax.set_title(m)
    fig.tight_layout(); fig.savefig(f"{out_dir}/Figure_confusion_{cond}.png", dpi=300); plt.close(fig)

def fig_percipher(out_dir, conds):
    plt = _plt(); fig, axs = plt.subplots(1, len(conds), figsize=(4.3 * len(conds), 3.2), sharey=True, squeeze=False)
    for ax, c in zip(axs[0], conds):
        for j, m in enumerate(MODELS):
            f = f"{out_dir}/{c}/Table_5_2_per_cipher_{m}.csv"
            if os.path.exists(f): ax.bar(np.arange(4) + (j - (len(MODELS) - 1) / 2) * (0.8 / len(MODELS)), pd.read_csv(f).F1.values, 0.8 / len(MODELS), color=COL[m], label=m)
        ax.set_xticks(range(4)); ax.set_xticklabels(CIPHERS, fontsize=8); ax.set_title(c); ax.set_ylim(60, 101)
    axs[0][0].set_ylabel("F1 (%)"); axs[0][0].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(f"{out_dir}/Figure_per_cipher_F1.png", dpi=300); plt.close(fig)

def fig_recall_heat(D, out_dir, cond):
    plt = _plt(); ci = D["cipher"]; r = ci[ci.Condition == cond].groupby(["Model", "Cipher", "Category"]).Recall.mean()
    fig, axs = plt.subplots(1, len(MODELS), figsize=(2.6 * len(MODELS), 2.7))
    for ax, m in zip(axs, MODELS):
        t = r.loc[m].unstack("Category")[CATEGORIES].reindex(CIPHERS); ax.imshow(t.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        for i in range(4):
            for j in range(3): ax.text(j, i, f"{t.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(3)); ax.set_xticklabels(CATEGORIES); ax.set_yticks(range(4)); ax.set_yticklabels(CIPHERS if m == MODELS[0] else [""] * 4); ax.set_title(m)
    fig.tight_layout(); fig.savefig(f"{out_dir}/Figure_recall_by_length_{cond}.png", dpi=300); plt.close(fig)

def fig_training(out_dir, cond):
    f = f"{out_dir}/{cond}/training_history.json"
    if not os.path.exists(f): return
    plt = _plt(); h = json.load(open(f)); fig, axs = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
    for ax, m in zip(axs, ["CNN", "BiLSTM"]):
        e = np.arange(1, len(h[m]["accuracy"]) + 1); ax.plot(e, 100 * np.array(h[m]["accuracy"]), "o-", label="Training"); ax.plot(e, 100 * np.array(h[m]["val_accuracy"]), "s--", label="Validation")
        ax.set_title(f"{m} ({cond})"); ax.set_xlabel("Epoch"); ax.legend(frameon=False)
    axs[0].set_ylabel("Accuracy (%)"); fig.tight_layout(); fig.savefig(f"{out_dir}/Figure_training_{cond}.png", dpi=300); plt.close(fig)

def fig_unseen_recall(D, out_dir):
    un = D["unseen"]
    if un.empty: return
    plt = _plt(); cs = sorted(un.Condition.unique()); fig, axs = plt.subplots(1, len(cs), figsize=(6 * len(cs), 3.2), sharey=True, squeeze=False)
    for ax, c in zip(axs[0], cs):
        g = un[un.Condition == c].groupby("Model").mean(numeric_only=True)
        for j, m in enumerate([m for m in MODELS if m in g.index]): ax.bar(np.arange(4) + (j - (len(MODELS) - 1) / 2) * (0.8 / len(MODELS)), [g.loc[m, f"Recall_{x}"] for x in CIPHERS], 0.8 / len(MODELS), color=COL[m], label=m)
        ax.set_xticks(range(4)); ax.set_xticklabels(CIPHERS); ax.set_title(f"{c} models, unseen keys")
    axs[0][0].set_ylabel("Recall (%)"); axs[0][0].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(f"{out_dir}/Figure_unseen_key_recall.png", dpi=300); plt.close(fig)

def _dumbbell(rows, fn, xlabel, labels, ref=None):
    plt = _plt(); BL, GR = "#2a6fdb", "#b9b9ad"; fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, (m, a, b, hot) in enumerate(rows):
        y = len(rows) - 1 - i; c = BL if hot else GR
        ax.plot([a, b], [y, y], color=c, lw=3, zorder=1); ax.scatter([a], [y], s=70, facecolor=c, alpha=.35, edgecolor=c, zorder=2); ax.scatter([b], [y], s=70, color=c, zorder=2)
        ax.text(a - 0.8, y, f"{a:.1f}", ha="right", va="center", fontsize=9); ax.text(b + 0.8, y, f"{b:.1f}", ha="left", va="center", fontsize=9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1]); ax.set_xlim(min(65, min(min(r[1], r[2]) for r in rows) - 6), 104); ax.set_ylim(-0.6, len(rows) - 0.3); ax.set_xlabel(xlabel)
    if ref: ax.axvline(ref, color="#777", ls="--", lw=1); ax.text(ref, len(rows) - 0.42, "95% threshold", ha="center", fontsize=8, color="#555")
    ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=GR, alpha=.5, label=labels[0]), plt.Line2D([], [], marker="o", ls="", color=GR, label=labels[1])],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(fn, dpi=300); plt.close(fig)

def fig_dumbbells(D, out_dir):
    ln, un = D["length"], D["unseen"]; N = {"Random Forest": "RF (TF-IDF)"}
    if "raw_random" in ln.Condition.values:
        g = ln[ln.Condition == "raw_random"].groupby(["Model", "Category"]).Accuracy.mean().unstack()
        rows = sorted([(N.get(m, m), g.loc[m, "Short"], g.loc[m, "Long"], m in PROPOSED) for m in MODELS if m in g.index], key=lambda r: -r[1])
        _dumbbell(rows, f"{out_dir}/Figure_short_vs_long_raw_random.png", "Accuracy (%), raw_random, mean over seeds", ("Short (10–50 characters)", "Long (201–500 characters)"), 95)
    if not un.empty and "raw_fixed" in un.Condition.values:
        g = un[un.Condition == "raw_fixed"].groupby("Model").mean(numeric_only=True)
        rows = sorted([(N.get(m, m), g.loc[m, "Acc_unseen"], g.loc[m, "Acc_seen"], m in PROPOSED) for m in MODELS if m in g.index], key=lambda r: r[1] - r[2])
        _dumbbell(rows, f"{out_dir}/Figure_seen_vs_unseen_keys_raw_fixed.png", "Accuracy (%), raw_fixed models, mean over seeds", ("Unseen keys", "Seen keys"))

def fig_method(out_dir):
    plt = _plt(); from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.6, 4.4)); ax.set_xlim(0, 112); ax.set_ylim(0, 60); ax.axis("off"); BL, LB, GR = "#2a6fdb", "#e8f0fc", "#555555"
    boxes = [("1  Data", ["Enron corpus:", "517,401 messages", "Clean text; remove", "exact and near-", "duplicates (MinHash)", "Balance by length", "(short/medium/long)", "66,009 messages"]),
             ("2  Encryption", ["4 ciphers:", "Caesar, Vigenère,", "Playfair, Rail Fence", "3 conditions:", "raw_fixed", "raw_random", "norm_fixed", "264,036 texts each"]),
             ("3  Split", ["Group-aware 80/20:", "4 encryptions of a", "message stay together", "Test: 52,808 texts", "3 seeds (42–44)", "Unseen-key test:", "fixed-key models,", "random-key texts"]),
             ("4  Models", ["Baselines: RF (TF-IDF),", "RF-Stat, RF-Stat+,", "CNN, BiLSTM", "Proposed:", "RF-CAF (attack-", "success features)", "CAF-Net (gated", "hybrid fusion)"]),
             ("5  Evaluation", ["Accuracy, F1, AUC", "McNemar, Wilson CI", "By length and cipher", "Minimum reliable", "length (≥ 95%)", "SHAP and LIME", "Unseen keys", "Conformal sets"])]
    w, gap, x0, y0, h = 19, 3.6, 1.2, 15, 42
    for i, (t, lines) in enumerate(boxes):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.2,rounding_size=1.2", fc=LB, ec=BL, lw=1.2))
        ax.add_patch(FancyBboxPatch((x, y0 + h - 6.5), w, 6.5, boxstyle="round,pad=0.2,rounding_size=1.2", fc=BL, ec=BL, lw=1.2))
        ax.text(x + w / 2, y0 + h - 3.3, t, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        for j, l in enumerate(lines): ax.text(x + 1.2, y0 + h - 10.2 - j * 3.9, l, ha="left", va="center", fontsize=5.9, color="#222")
        if i < 4: ax.add_patch(FancyArrowPatch((x + w + 0.5, y0 + h / 2), (x + w + gap - 0.5, y0 + h / 2), arrowstyle="-|>", mutation_scale=10, color=GR, lw=1.3))
    ax.text(56, 11.2, "Research questions answered by stage 5", ha="center", fontsize=7.5, color=GR, fontweight="bold")
    for i, (a, b) in enumerate([("RQ1", "overall accuracy"), ("RQ2", "message length"), ("RQ3", "SHAP, LIME, gate"), ("RQ4", "unseen keys"), ("RQ5", "proposed CAF-Net")]):
        x = 2.5 + i * 21.8; ax.add_patch(FancyBboxPatch((x, 2.2), 20, 5.2, boxstyle="round,pad=0.2,rounding_size=1", fc="white", ec=BL, lw=1)); ax.text(x + 10, 4.8, f"{a}: {b}", ha="center", va="center", fontsize=6.2)
    fig.savefig(f"{out_dir}/Figure_method_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# ------------------------------------------------------------------------------
# 10. Utilities: demo corpus for smoke tests, quick configuration
# ------------------------------------------------------------------------------
def make_demo_emails(path, n=2500, seed=1):
    """Synthetic e-mail-like corpus (header + blank line + body) used only to test that the pipeline runs end to end."""
    rng = np.random.RandomState(seed)
    words = ("please send the report before the meeting tomorrow we need to review the contract schedule call me when you are free "
             "thanks for the update let us discuss the budget and the plan for next week the team will join at noon I will forward the "
             "document later today could you confirm the numbers regarding the project deadline good morning see you soon").split()
    rows = []
    for i in range(n):
        k = int(rng.choice([6, 25, 70])); body = " ".join(rng.choice(words, size=k + int(rng.randint(0, 12))))
        body = body.capitalize() + ("." if rng.rand() < 0.7 else "") + (f" Ref {int(rng.randint(100, 999))}, call {int(rng.randint(1000, 9999))}." if rng.rand() < 0.4 else "")
        rows.append((f"user/{i}", f"Message-ID: <{i}>\nFrom: a@b.com\nSubject: s{i}\n\n{body}"))
    pd.DataFrame(rows, columns=["file", "message"]).to_csv(path, index=False)

def quick_cfg(cfg):
    cfg = dict(cfg); cfg.update(tfidf_features=3000, rf_trees_tfidf=15, rf_trees_stat=30, epochs=2, caf_epochs=3, shap_n_stat=40, shap_n_tfidf=20, lime_samples=100); return cfg

# ==============================================================================
# PAPER DRIVER: experiments and figures of the journal paper
#   "Classic Cipher Classification in Encrypted Emails: The Impact of Text Length on
#    Deep Learning Models with Explainable AI"
#   - conditions: raw_fixed, raw_random, norm_fixed (add norm_random with --conditions)
#   - seeds 42-44 (use --seeds 42 for a single seed)
#   - writes Tables 1-8 (CSV) and Figures 1-5 (PNG) to --out
#   - Tables 7-8 and Figures 4-5 belong to the proposed method (CAF-Net)
# Usage (Colab):  !python paper_code.py --emails /content/emails.csv --out paper_results
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emails"); ap.add_argument("--out", default="results_paper")
    ap.add_argument("--conditions", nargs="+", default=["raw_fixed", "raw_random", "norm_fixed"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--nrows", type=int, default=None); ap.add_argument("--quick", action="store_true"); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--ablation", action="store_true", help="also train the three CAF-Net ablation variants")
    a = ap.parse_args(); cfg = quick_cfg(DEFAULT_CFG) if a.quick else dict(DEFAULT_CFG); cfg["ablation"] = a.ablation
    self_test(); os.makedirs(a.out, exist_ok=True); random.seed(42); np.random.seed(42)
    if a.demo: a.emails = f"{a.out}/demo_emails.csv"; make_demo_emails(a.emails)
    df = build_dataset(a.emails, a.out, seed=cfg["seed_data"], nrows=a.nrows)           # Table 1 = Table_3_2_dataset_pipeline_counts.csv
    for cond in a.conditions: run_condition(cond, a.seeds, df, cfg, a.out)
    D = load_runs(a.out); T = aggregate_tables(D, a.out, a.conditions)
    # Paper tables: Table 3 overall accuracy, Table 4 by length, Table 5 minimum length, Table 6 unseen keys
    for src, dst in (("Table_overall_accuracy", "Table_3"), ("Table_accuracy_by_length", "Table_4"), ("Table_min_reliable_length", "Table_5"), ("Table_unseen_keys", "Table_6"),
                     ("Table_conformal_CAF_Net", "Table_7"), ("Table_ablation_CAF_Net", "Table_8")):
        if os.path.exists(f"{T}/{src}.csv"): os.replace(f"{T}/{src}.csv", f"{T}/{dst}.csv")
    fig_method(a.out)                                                                    # Figure 1
    fig_dumbbells(D, a.out)                                                              # Figure 2 (length) and Figure 3 (unseen keys)
    fig_gate(D, a.out, a.conditions); fig_conformal(D, a.out, a.conditions)                 # Figures 4-5 (proposed method)
    for c in a.conditions: fig_confusion(a.out, c)
    print("finished. Tables:", T, "| Figures:", a.out)

if __name__ == "__main__":
    main()
