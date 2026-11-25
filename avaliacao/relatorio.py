"""Exporta métricas calculadas exclusivamente a partir de julgamentos salvos."""
import csv
from collections import defaultdict
from statistics import mean

from .experimentos import classification
from .judge import LABELS


def table(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot(path, names, panels):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(max(6, len(names) * 1.45), 4 * len(panels)), layout="constrained")
    for ax, (title, values) in zip(fig.subplots(len(panels), 1, squeeze=False)[:, 0], panels):
        ax.bar(names, values)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", labelrotation=30)
    fig.savefig(path, dpi=140)


def report(task, items, output):
    groups = defaultdict(list)
    for item in items:
        pairs = [tuple(pair) for pair in item["pairs"]]
        groups["total"].extend(pairs)
        if task == "mtbench":
            groups[f"turno {item['turn']}"].extend(pairs)
    metrics, classes, confusion = [], [], []
    for group, pairs in groups.items():
        summary, details, counts = classification(task, pairs)
        metrics.append(dict(grupo=group, **summary))
        classes.extend(dict(grupo=group, **r) for r in details)
        for gold in LABELS[task]:
            for label in (*LABELS[task], "invalid"):
                confusion.append(dict(grupo=group, referencia=gold, juiz=label, n=counts[gold, label]))
    table(output / "metricas.csv", metrics)
    table(output / "classes.csv", classes)
    table(output / "confusao.csv", confusion)
    plot(output / "metricas.png", [r["grupo"] for r in metrics],
         [(key, [r[key] for r in metrics]) for key in ("acuracia", "macro_f1", "cobertura")])
    if task == "gsm8k":
        strategies = defaultdict(list)
        for item in items:
            for name, result in item["strategies"].items():
                strategies[name].append(result)
        summary = []
        for name, rows in strategies.items():
            summary.append(dict(estrategia=name, problemas=len(rows), acuracia=mean(r["acerto"] for r in rows),
                                amostras_media=mean(r["amostras"] for r in rows),
                                economia=1-sum(r["amostras"] for r in rows)/sum(r["limite"] for r in rows),
                                paradas_erradas=sum(r["amostras"] < r["limite"] and not r["acerto"]
                                                    for r in rows) if name.startswith("adaptativa") else "",
                                sem_resposta=sum(r["resposta"] is None for r in rows)))
        table(output / "estrategias.csv", summary)
        plot(output / "estrategias.png", [r["estrategia"] for r in summary],
             [(key, [r[key] for r in summary]) for key in ("acuracia", "amostras_media")])
