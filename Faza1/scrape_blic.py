# Faza 1: Blic.rs -> CSV/JSON po kategoriji (tačne kolone iz PDF-a)
import os, re, json, csv, time
import requests, feedparser, pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from dateutil import parser as dtp

# --- Podesi ovo ---
INIT = "LM"  # tvoja dva inicijala
FILE_PREFIX = "blic"  # ako baš traže, promijeni u "buka"
# ------------------

OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)
PORTAL = "blic.rs"

# Kategorije -> RSS (lako i stabilno)
RSS_MAP = {
    "Vesti": "https://www.blic.rs/rss/vesti.xml",
    "Politika": "https://www.blic.rs/rss/politika.xml",
    "Svet": "https://www.blic.rs/rss/svet.xml",
    "Društvo": "https://www.blic.rs/rss/drustvo.xml",
    "Beograd": "https://www.blic.rs/rss/beograd.xml",
    "Biznis": "https://www.blic.rs/rss/biznis.xml",
    "Zabava": "https://www.blic.rs/rss/zabava.xml",
    "Kultura": "https://www.blic.rs/rss/kultura.xml",
}

UA = {"User-Agent": "Mozilla/5.0"}

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def parse_article(url: str):
    r = requests.get(url, headers=UA, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # naslov
    h1 = soup.find("h1")
    title = clean(h1.get_text()) if h1 else ""

    # datum (prvo meta, pa <time>)
    date_text = ""
    for m in soup.find_all("meta"):
        name = (m.get("name") or m.get("property") or "").lower()
        if any(k in name for k in ["publish", "date", "time"]):
            date_text = m.get("content") or ""
            if date_text: break
    if not date_text:
        t = soup.find("time")
        date_text = t.get("datetime") if t and t.get("datetime") else (t.get_text() if t else "")

    try:
        date_iso = dtp.parse(date_text, fuzzy=True).date().isoformat() if date_text else ""
    except Exception:
        date_iso = ""

    # tekst (spoji duže <p>)
    body = soup.find("article") or soup
    paras = []
    for p in body.find_all("p"):
        txt = clean(p.get_text())
        if len(txt) > 40 and not txt.lower().startswith(("pročitajte", "video:", "foto:")):
            paras.append(txt)
    text = " ".join(paras)
    return date_iso, title, text

def write_outputs(category: str, rows: list):
    cols = ["Portal","Kategorija","ID","Url","Datum","Naslov","Tekst",
            "Ekstraktivna_sumarizacija","Apstraktivna_sumarizacija",
            "ChatGPT_sumarizacija","Gemini_sumarizacija","Claude_sumarizacija"]

    base = f"{FILE_PREFIX}_{category.lower().replace(' ', '_')}_{INIT}"
    csv_path = os.path.join(OUT_DIR, f"{base}.csv")
    json_path = os.path.join(OUT_DIR, f"{base}.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows: w.writerow(r)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("[OK]", category, "->", csv_path, "|", json_path)

def main():
    idx = 0
    for cat, rss in RSS_MAP.items():
        feed = feedparser.parse(rss)
        rows = []
        for e in tqdm(feed.entries, desc=cat):
            url = getattr(e, "link", "")
            if not url: continue
            try:
                d, t, txt = parse_article(url)
                if len(txt.split()) < 60 or not t:  # minimalni kvalitet
                    continue
                rid = f"{idx:05d}"; idx += 1
                rows.append({
                    "Portal": PORTAL, "Kategorija": cat, "ID": rid,
                    "Url": url, "Datum": d, "Naslov": t, "Tekst": txt,
                    "Ekstraktivna_sumarizacija": "", "Apstraktivna_sumarizacija": "",
                    "ChatGPT_sumarizacija": "", "Gemini_sumarizacija": "", "Claude_sumarizacija": ""
                })
                time.sleep(0.2)
            except Exception as ex:
                print("WARN:", ex)
        if rows: write_outputs(cat, rows)

if __name__ == "__main__":
    main()
