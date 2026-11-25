"""Execute python -m avaliacao --help na raiz do projeto."""
import argparse
from contextlib import closing
import hashlib
from itertools import islice
import json
from pathlib import Path
import sqlite3

from .experimentos import gsm, mt
from .judge import MODEL, REVISION, QwenJudge, messages_for, parse
from .relatorio import report

DATA = "lmsys/mt_bench_human_judgments"
DATA_REV = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def records(path):
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run(task, rows, output, source, judge, limit=None):
    output.mkdir(parents=True, exist_ok=True)
    config = json.dumps(dict(task=task, source=source, model=MODEL, revision=REVISION,
                             protocol=1, quant="nf4-fp16", limit=limit), sort_keys=True)
    with closing(sqlite3.connect(output / "checkpoint.sqlite", timeout=60)) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, raw TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, result TEXT NOT NULL);
        """)
        with db:
            db.execute("INSERT OR IGNORE INTO config VALUES (1,?)", (config,))
            if db.execute("SELECT value FROM config").fetchone()[0] != config:
                raise ValueError("Entrada ou configuração mudou. Escolha outra pasta de saída.")

        def ask(payload):
            messages = messages_for(task, payload)
            key = digest(json.dumps(messages, sort_keys=True).encode())
            with db:
                db.execute("BEGIN IMMEDIATE")
                cached = db.execute("SELECT raw FROM calls WHERE key=?", (key,)).fetchone()
                if cached:
                    raw = cached[0]
                else:
                    raw = judge(messages)
                    db.execute("INSERT INTO calls VALUES (?,?)", (key, raw))
            return parse(task, raw)

        seen = set()
        for index, row in enumerate(islice(rows, limit)):
            identifier = row["indice"] if task == "gsm8k" else index
            if identifier in seen:
                raise ValueError(f"Identificador duplicado na entrada: {identifier}")
            seen.add(identifier)
            if db.execute("SELECT 1 FROM items WHERE id=?", (identifier,)).fetchone():
                continue
            item = (gsm if task == "gsm8k" else mt)(row, ask)
            with db:
                db.execute("INSERT OR IGNORE INTO items VALUES (?,?)", (identifier, json.dumps(item)))
            if (index + 1) % 25 == 0:
                print(f"{index + 1} registros concluídos.", flush=True)
        if not seen:
            raise ValueError("Entrada vazia.")
        items = [json.loads(r[0]) for r in db.execute("SELECT result FROM items ORDER BY id")]
        report(task, items, output)
    print(f"Relatório disponível em {output}")


def main():
    parser = argparse.ArgumentParser(description="Avaliação incremental com Qwen3-8B em GPU")
    parser.add_argument("experimento", choices=["gsm8k", "mtbench"])
    parser.add_argument("--entrada", type=Path, help="JSONL local no formato do benchmark")
    parser.add_argument("--saida", type=Path, help="Diretório exclusivo para este experimento")
    parser.add_argument("--limite", type=int, help="Número máximo de problemas ou anotações")
    args = parser.parse_args()
    if args.limite is not None and args.limite < 1:
        parser.error("--limite deve ser positivo")
    path = args.entrada
    if args.experimento == "gsm8k" and path is None:
        path = Path("respostas.jsonl")
    if path is None:
        from datasets import load_dataset
        rows = load_dataset(DATA, revision=DATA_REV, split="human", streaming=True)
        source = f"{DATA}@{DATA_REV}:human"
    else:
        source = file_digest(path)
        rows = records(path)
    run(args.experimento, rows, args.saida or Path("saidas_juiz") / args.experimento,
        source, QwenJudge(), args.limite)


if __name__ == "__main__":
    main()
