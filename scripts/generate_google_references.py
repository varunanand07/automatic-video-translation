import os
import argparse
import time
import random
import re
import json

try:
    from deep_translator import GoogleTranslator as Deep
except Exception:
    Deep = None
try:
    from googletrans import Translator as GTrans
except Exception:
    GTrans = None

def chunks(s, limit):
    parts = re.split(r'(?<=[.!?])\s+|\n+', s)
    res, cur = [], ''
    for p in (p.strip() for p in parts if p.strip()):
        p += ' '
        if len(p) > limit:
            if cur: res.append(cur.strip()); cur = ''
            for i in range(0, len(p), limit):
                seg = p[i:i+limit].strip()
                if seg: res.append(seg)
            continue
        if len(cur) + len(p) <= limit:
            cur += p
        else:
            res.append(cur.strip()); cur = p
    if cur: res.append(cur.strip())
    return res

def translate_unit(t, text, use_deep):
    for i in range(3):
        try:
            time.sleep(random.uniform(1, 2))
            return t.translate(text) if use_deep else t.translate(text, src='en', dest='es').text
        except Exception:
            if i < 2: time.sleep((2**i) + random.uniform(1, 3))
    return None

def translate_text(t, text, use_deep):
    for limit in (4500, 3000, 2000):
        out, ok = [], True
        for c in chunks(text, limit):
            tr = translate_unit(t, c, use_deep)
            if not tr: ok = False; break
            out.append(tr)
        if ok: return ' '.join(out)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    base = f"data/{args.dataset}"
    src, dst = f"{base}/original_transcripts", f"{base}/reference_translations"
    os.makedirs(dst, exist_ok=True)

    if Deep: t, use_deep = Deep(source='en', target='es'), True
    elif GTrans: t, use_deep = GTrans(), False
    else: raise RuntimeError("Install deep-translator or googletrans")

    with open(f"{base}/metadata.json", 'r', encoding='utf-8') as f:
        meta = json.load(f)

    saved = 0
    for item in meta:
        vid = item['id']
        inp, outp = f"{src}/{vid}.txt", f"{dst}/{vid}_es.txt"
        if not os.path.exists(inp):
            continue
        if os.path.exists(outp) and not args.overwrite:
            saved += 1; continue
        with open(inp, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if not text:
            continue
        tr = translate_text(t, text, use_deep)
        if not tr:
            continue
        with open(outp, 'w', encoding='utf-8') as f:
            f.write(tr)
        saved += 1
        time.sleep(random.uniform(2, 4))
    print(f"{saved} reference translation files saved to {dst}")

if __name__ == '__main__':
    main()