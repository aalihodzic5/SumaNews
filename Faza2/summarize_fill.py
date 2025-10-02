# Faza 2: Popuni Ekstraktivna_ i Apstraktivna_ kolone; AI kolone ostaju za ručno popunjavanje
import os, glob, pandas as pd
from tqdm import tqdm
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from nltk.tokenize import sent_tokenize

def extractive(text: str, n: int = 3) -> str:
    if not text or len(text.split()) < 60: return ""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summ = TextRankSummarizer()
    sents = [str(s) for s in summ(parser.document, n)]
    return " ".join(sents)

def abstractive(text: str) -> str:
    if not text or len(text.split()) < 60: return ""
    s = sent_tokenize(text)
    # jednostavan parafrazni sažetak: prve 2–3 rečenice (lagani rez)
    return " ".join(s[:2])

for path in glob.glob("out/*_*.csv"):
    df = pd.read_csv(path)
    need_save = False
    if "Ekstraktivna_sumarizacija" not in df.columns: df["Ekstraktivna_sumarizacija"] = ""
    if "Apstraktivna_sumarizacija" not in df.columns: df["Apstraktivna_sumarizacija"] = ""

    for i, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(path)):
        text = str(row.get("Tekst", ""))
        if not str(row.get("Ekstraktivna_sumarizacija","")).strip():
            df.at[i, "Ekstraktivna_sumarizacija"] = extractive(text)
            need_save = True
        if not str(row.get("Apstraktivna_sumarizacija","")).strip():
            df.at[i, "Apstraktivna_sumarizacija"] = abstractive(text)
            need_save = True

    if need_save:
        df.to_csv(path, index=False, encoding="utf-8")
        print("[OK] Ažurirano:", path)
