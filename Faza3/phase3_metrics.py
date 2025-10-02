import argparse, os, re, glob, warnings
import pandas as pd
import numpy as np
from itertools import islice
from rouge_score import rouge_scorer

# --- BERTScore: opcionalno ---
_HAS_BERT = True
try:
    from bert_score import score as bertscore
except Exception:
    _HAS_BERT = False

# Utišaj BERT warninge o baseline-u:
warnings.filterwarnings("ignore", message="Baseline not Found*")

# ------------------ Tokenizacija i stop-riječi (bez NLTK download-a) ------------------
_BARE_STOPWORDS = set("""
i u na za da je nije su sam smo ste će ovo ono ali kao koji koja koje kojeg kojoj kojih kojim kojima
od do po pri s sa bez kroz između prema vrlo više manje već još tada sada kad kada jer dok
the a an and or of to in on for with by from as is are was were be been being this that these those
""".split())

def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def toks(text: str):
    # Jednostavna UNICODE tokenizacija bez punkt-a
    return re.findall(r"\w+", norm(text), flags=re.UNICODE)

def important_words(text: str):
    return {w for w in toks(text) if w not in _BARE_STOPWORDS and len(w) > 2}

def ngrams(seq, n):
    it = iter(seq)
    window = list(islice(it, n))
    if len(window) == n:
        yield tuple(window)
    for e in it:
        window = window[1:] + [e]
        yield tuple(window)

# ------------------ Metrike ------------------
def compression_ratio(original: str, summary: str):
    ow = len(toks(original)) or 1
    sw = len(toks(summary)) or 1
    return ow / sw

def coverage_score(original: str, summary: str):
    imp = important_words(original)
    if not imp:
        return 0.0
    summ_set = set(toks(summary))
    present = sum(1 for w in imp if w in summ_set)
    return present / len(imp)

def density_score(original: str, summary: str):
    ot = toks(original)
    st = toks(summary)
    if not st:
        return 0.0
    orig_bi = set(ngrams(ot, 2))
    count_in_bi = 0
    i = 0
    while i < len(st) - 1:
        if (st[i], st[i+1]) in orig_bi:
            count_in_bi += 2
            i += 2
        else:
            i += 1
    return min(1.0, count_in_bi / max(1, len(st)))

def rouge_all(candidate: str, reference: str):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=False)
    scores = scorer.score(reference, candidate)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

def safe_bertscore(cands, refs, lang_code, enabled=True):
    if not enabled or not _HAS_BERT:
        return [None for _ in cands]
    try:
        # brže i bez baseline fajlova
        P, R, F1 = bertscore(
            cands, refs,
            lang=lang_code,
            rescale_with_baseline=False,
            model_type="bert-base-multilingual-cased",
            device="cpu",
            verbose=False,
            idf=False
        )
        return F1.numpy().tolist()
    except Exception:
        return [None for _ in cands]

# ------------------ Kolone i obrada ------------------
def find_col(df, candidates):
    # prvo tačno ime
    for c in candidates:
        if c in df.columns:
            return c
    # onda regex fuzzy
    for col in df.columns:
        for c in candidates:
            pat = c.replace('*','.*')
            if re.fullmatch(pat, col, flags=re.IGNORECASE):
                return col
    return None

def detect_columns(df):
    orig_col = find_col(df, ['Originalni_tekst','Tekst','original*','tekst'])
    kat_col  = find_col(df, ['Kategorija','kategorija','Category','category'])
    id_col   = find_col(df, ['ID','Id','id'])

    model_map = {
        'Ekstraktivna': find_col(df, ['Ekstraktivna_sumarizacija','ekstraktivna*','Ekstraktivna']),
        'Apstraktivna': find_col(df, ['Apstraktivna_sumarizacija','apstraktivna*','Apstraktivna']),
        'ChatGPT':      find_col(df, ['ChatGPT_sumarizacija','chatgpt*','GPT*','gpt*']),
        'Gemini':       find_col(df, ['Gemini_sumarizacija','gemini*']),
        'Claude':       find_col(df, ['Claude_sumarizacija','claude*']),
    }
    return orig_col, kat_col, id_col, model_map

def pick_reference(row, model_map):
    # preferiraj apstraktivnu -> ekstraktivnu -> ChatGPT -> Gemini -> Claude -> original
    for k in ['Apstraktivna','Ekstraktivna','ChatGPT','Gemini','Claude']:
        col = model_map.get(k)
        if col:
            val = str(row.get(col, "") or "")
            if val.strip():
                return val
    return None

def process_file(path, lang_code='hr', use_bert=True):
    df = pd.read_csv(path)
    orig_col, kat_col, id_col, model_map = detect_columns(df)

    if not orig_col or not id_col:
        print(f"[WARN] Preskačem {os.path.basename(path)} – nema ključnih kolona (Originalni_tekst/Tekst ili ID).")
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        original = str(row.get(orig_col, "") or "")
        kategorija = str(row.get(kat_col, "") or "")
        _id = str(row.get(id_col, "") or "")

        ref = pick_reference(row, model_map) or original

        cand_texts, ref_texts, metas = [], [], []
        for model_name, col in model_map.items():
            if not col:
                continue
            summary = str(row.get(col, "") or "").strip()
            if not summary:
                continue

            comp = compression_ratio(original, summary)
            cov  = coverage_score(original, summary)
            dens = density_score(original, summary)
            r1, r2, rL = rouge_all(summary, ref)

            cand_texts.append(summary)
            ref_texts.append(ref)
            metas.append({
                "Kategorija": kategorija,
                "ID": _id,
                "Model": model_name,
                "Compression": round(comp, 4),
                "Coverage": round(cov, 4),
                "Density": round(dens, 4),
                "ROUGE-1": round(r1, 4),
                "ROUGE-2": round(r2, 4),
                "ROUGE-L": round(rL, 4),
            })

        if cand_texts:
            f1s = safe_bertscore(cand_texts, ref_texts, lang_code, enabled=use_bert)
            for meta, f1 in zip(metas, f1s):
                meta["BERTScore"] = round(float(f1), 4) if f1 is not None else ""
                rows.append(meta)

    return pd.DataFrame(rows)

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="./out", help="Folder sa CSV-ovima iz Faze 2")
    ap.add_argument("--out_all", default="./out/metrics_ALL.csv", help="Objedinjeni izlazni CSV")
    ap.add_argument("--per_file", action="store_true", help="Snimi i metrics_* po fajlu")
    ap.add_argument("--lang", default="hr", help="Jezik za BERTScore (hr/sr/en...)")
    ap.add_argument("--no-bert", action="store_true", help="Isključi BERTScore")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.in_dir, "blic_*_LM.csv")))
    if not csvs:
        print(f"[ERR] Nema ulaznih fajlova u {args.in_dir} (očekujem blic_*_LM.csv).")
        raise SystemExit(1)

    use_bert = (not args.no_bert) and _HAS_BERT
    if not _HAS_BERT and not args.no_bert:
        print("[INFO] bert-score nije instaliran ili nije dostupan – BERTScore će biti prazan.")

    all_parts = []
    for p in csvs:
        print("[INFO] Obrada:", os.path.basename(p))
        dfm = process_file(p, lang_code=args.lang, use_bert=use_bert)
        if dfm.empty:
            print("   [WARN] Nema metrika za", os.path.basename(p))
            continue
        all_parts.append(dfm)
        if args.per_file:
            out_p = os.path.join(args.in_dir, f"metrics_{os.path.splitext(os.path.basename(p))[0]}.csv")
            dfm.to_csv(out_p, index=False)
            print("   ->", out_p)

    if not all_parts:
        print("[ERR] Nije generisan nijedan izlaz.")
        raise SystemExit(2)

    out_all = pd.concat(all_parts, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_all), exist_ok=True)
    out_all.to_csv(args.out_all, index=False)
    print("[OK] Sačuvano:", args.out_all)

if __name__ == "__main__":
    main()


import argparse, os, re, glob, warnings
import pandas as pd
import numpy as np
from itertools import islice
from rouge_score import rouge_scorer

# --- BERTScore: opcionalno ---
_HAS_BERT = True
try:
    from bert_score import score as bertscore
except Exception:
    _HAS_BERT = False

# Utišaj BERT warninge o baseline-u:
warnings.filterwarnings("ignore", message="Baseline not Found*")

# ------------------ Tokenizacija i stop-riječi (bez NLTK download-a) ------------------
_BARE_STOPWORDS = set("""
i u na za da je nije su sam smo ste će ovo ono ali kao koji koja koje kojeg kojoj kojih kojim kojima
od do po pri s sa bez kroz između prema vrlo više manje već još tada sada kad kada jer dok
the a an and or of to in on for with by from as is are was were be been being this that these those
""".split())

def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def toks(text: str):
    # Jednostavna UNICODE tokenizacija bez punkt-a
    return re.findall(r"\w+", norm(text), flags=re.UNICODE)

def important_words(text: str):
    return {w for w in toks(text) if w not in _BARE_STOPWORDS and len(w) > 2}

def ngrams(seq, n):
    it = iter(seq)
    window = list(islice(it, n))
    if len(window) == n:
        yield tuple(window)
    for e in it:
        window = window[1:] + [e]
        yield tuple(window)

# ------------------ Metrike ------------------
def compression_ratio(original: str, summary: str):
    ow = len(toks(original)) or 1
    sw = len(toks(summary)) or 1
    return ow / sw

def coverage_score(original: str, summary: str):
    imp = important_words(original)
    if not imp:
        return 0.0
    summ_set = set(toks(summary))
    present = sum(1 for w in imp if w in summ_set)
    return present / len(imp)

def density_score(original: str, summary: str):
    ot = toks(original)
    st = toks(summary)
    if not st:
        return 0.0
    orig_bi = set(ngrams(ot, 2))
    count_in_bi = 0
    i = 0
    while i < len(st) - 1:
        if (st[i], st[i+1]) in orig_bi:
            count_in_bi += 2
            i += 2
        else:
            i += 1
    return min(1.0, count_in_bi / max(1, len(st)))

def rouge_all(candidate: str, reference: str):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=False)
    scores = scorer.score(reference, candidate)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

def safe_bertscore(cands, refs, lang_code, enabled=True):
    if not enabled or not _HAS_BERT:
        return [None for _ in cands]
    try:
        # brže i bez baseline fajlova
        P, R, F1 = bertscore(
            cands, refs,
            lang=lang_code,
            rescale_with_baseline=False,
            model_type="bert-base-multilingual-cased",
            device="cpu",
            verbose=False,
            idf=False
        )
        return F1.numpy().tolist()
    except Exception:
        return [None for _ in cands]

# ------------------ Kolone i obrada ------------------
def find_col(df, candidates):
    # prvo tačno ime
    for c in candidates:
        if c in df.columns:
            return c
    # onda regex fuzzy
    for col in df.columns:
        for c in candidates:
            pat = c.replace('*','.*')
            if re.fullmatch(pat, col, flags=re.IGNORECASE):
                return col
    return None

def detect_columns(df):
    orig_col = find_col(df, ['Originalni_tekst','Tekst','original*','tekst'])
    kat_col  = find_col(df, ['Kategorija','kategorija','Category','category'])
    id_col   = find_col(df, ['ID','Id','id'])

    model_map = {
        'Ekstraktivna': find_col(df, ['Ekstraktivna_sumarizacija','ekstraktivna*','Ekstraktivna']),
        'Apstraktivna': find_col(df, ['Apstraktivna_sumarizacija','apstraktivna*','Apstraktivna']),
        'ChatGPT':      find_col(df, ['ChatGPT_sumarizacija','chatgpt*','GPT*','gpt*']),
        'Gemini':       find_col(df, ['Gemini_sumarizacija','gemini*']),
        'Claude':       find_col(df, ['Claude_sumarizacija','claude*']),
    }
    return orig_col, kat_col, id_col, model_map

def pick_reference(row, model_map):
    # preferiraj apstraktivnu -> ekstraktivnu -> ChatGPT -> Gemini -> Claude -> original
    for k in ['Apstraktivna','Ekstraktivna','ChatGPT','Gemini','Claude']:
        col = model_map.get(k)
        if col:
            val = str(row.get(col, "") or "")
            if val.strip():
                return val
    return None

def process_file(path, lang_code='hr', use_bert=True):
    df = pd.read_csv(path)
    orig_col, kat_col, id_col, model_map = detect_columns(df)

    if not orig_col or not id_col:
        print(f"[WARN] Preskačem {os.path.basename(path)} – nema ključnih kolona (Originalni_tekst/Tekst ili ID).")
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        original = str(row.get(orig_col, "") or "")
        kategorija = str(row.get(kat_col, "") or "")
        _id = str(row.get(id_col, "") or "")

        ref = pick_reference(row, model_map) or original

        cand_texts, ref_texts, metas = [], [], []
        for model_name, col in model_map.items():
            if not col:
                continue
            summary = str(row.get(col, "") or "").strip()
            if not summary:
                continue

            comp = compression_ratio(original, summary)
            cov  = coverage_score(original, summary)
            dens = density_score(original, summary)
            r1, r2, rL = rouge_all(summary, ref)

            cand_texts.append(summary)
            ref_texts.append(ref)
            metas.append({
                "Kategorija": kategorija,
                "ID": _id,
                "Model": model_name,
                "Compression": round(comp, 4),
                "Coverage": round(cov, 4),
                "Density": round(dens, 4),
                "ROUGE-1": round(r1, 4),
                "ROUGE-2": round(r2, 4),
                "ROUGE-L": round(rL, 4),
            })

        if cand_texts:
            f1s = safe_bertscore(cand_texts, ref_texts, lang_code, enabled=use_bert)
            for meta, f1 in zip(metas, f1s):
                meta["BERTScore"] = round(float(f1), 4) if f1 is not None else ""
                rows.append(meta)

    return pd.DataFrame(rows)

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="./out", help="Folder sa CSV-ovima iz Faze 2")
    ap.add_argument("--out_all", default="./out/metrics_ALL.csv", help="Objedinjeni izlazni CSV")
    ap.add_argument("--per_file", action="store_true", help="Snimi i metrics_* po fajlu")
    ap.add_argument("--lang", default="hr", help="Jezik za BERTScore (hr/sr/en...)")
    ap.add_argument("--no-bert", action="store_true", help="Isključi BERTScore")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.in_dir, "blic_*_LM.csv")))
    if not csvs:
        print(f"[ERR] Nema ulaznih fajlova u {args.in_dir} (očekujem blic_*_LM.csv).")
        raise SystemExit(1)

    use_bert = (not args.no_bert) and _HAS_BERT
    if not _HAS_BERT and not args.no_bert:
        print("[INFO] bert-score nije instaliran ili nije dostupan – BERTScore će biti prazan.")

    all_parts = []
    for p in csvs:
        print("[INFO] Obrada:", os.path.basename(p))
        dfm = process_file(p, lang_code=args.lang, use_bert=use_bert)
        if dfm.empty:
            print("   [WARN] Nema metrika za", os.path.basename(p))
            continue
        all_parts.append(dfm)
        if args.per_file:
            out_p = os.path.join(args.in_dir, f"metrics_{os.path.splitext(os.path.basename(p))[0]}.csv")
            dfm.to_csv(out_p, index=False)
            print("   ->", out_p)

    if not all_parts:
        print("[ERR] Nije generisan nijedan izlaz.")
        raise SystemExit(2)

    out_all = pd.concat(all_parts, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_all), exist_ok=True)
    out_all.to_csv(args.out_all, index=False)
    print("[OK] Sačuvano:", args.out_all)

if __name__ == "__main__":
    main()
