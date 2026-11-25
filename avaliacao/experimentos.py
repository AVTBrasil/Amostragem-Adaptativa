"""Protocolos de avaliação sobre respostas já geradas."""
from collections import Counter
from math import isfinite

from scipy.stats import beta

from .judge import LABELS


def correct(value, gold):
    return value is not None and abs(value - gold) < 1e-6


def vote(counts):
    return counts.most_common(1)[0][0] if counts else None


def strategies(numbers, gold, suffix):
    # Mesma ordem, desempate e regra beta do notebook original.
    counts = Counter()
    results = {}
    for n, value in enumerate(numbers[:16], 1):
        if value is not None:
            counts[value] += 1
        prediction = vote(counts)
        def save(name):
            results[name + suffix] = dict(acerto=correct(prediction, gold), amostras=n,
                                          resposta=prediction, limite=min(16, len(numbers)))
        if n in (4, 8, 16):
            save(f"fixa {n}")
        top = counts.most_common(2)
        confidence = beta.sf(0.5, top[0][1] + 1, (top[1][1] if len(top) > 1 else 0) + 1) if top else 0
        for threshold in (0.90, 0.95):
            name = f"adaptativa {threshold:.2f}"
            if name + suffix not in results and (confidence >= threshold or n == min(16, len(numbers))):
                save(name)
    return results


def gsm(row, ask):
    gold = row["gabarito"]
    answers = row["respostas"]
    if not isfinite(gold) or not answers:
        raise ValueError("Gabarito inválido ou lista de respostas vazia.")
    numbers, approved, pairs = [], [], []
    for answer in answers:
        num = answer["numero"]
        if num is not None and not isfinite(num):
            raise ValueError("Número extraído inválido.")
        verdict = ask({"question": row["pergunta"], "candidate": answer["texto"]})
        numbers.append(num)
        approved.append(num if verdict == "correct" else None)
        pairs.append(("correct" if correct(num, gold) else "incorrect", verdict))
    results = strategies(numbers, gold, "")
    results.update(strategies(approved, gold, " + juiz"))
    return dict(pairs=pairs, strategies=results)


def mt(row, ask):
    turn = row["turn"]
    if turn not in (1, 2):
        raise ValueError(f"Turno desconhecido: {turn}")
    payload = {}
    for side in ("a", "b"):
        history = row["conversation_" + side][:2 * turn]
        if [x["role"] for x in history] != ["user", "assistant"] * turn:
            raise ValueError(f"Histórico inválido para o turno {turn}.")
        payload[side.upper()] = history
    gold = {"model_a": "A", "model_b": "B", "tie": "tie"}[row["winner"]]
    return dict(pairs=[(gold, ask(payload))], turn=turn,
                question_id=row.get("question_id"), model_a=row.get("model_a"),
                model_b=row.get("model_b"))


def classification(task, pairs):
    counts = Counter(pairs)
    n = sum(counts.values())
    labels = LABELS[task]
    rows = []
    for label in labels:
        tp = counts[label, label]
        predicted = sum(v for (g, p), v in counts.items() if p == label)
        support = sum(v for (g, p), v in counts.items() if g == label)
        rows.append(dict(classe=label, suporte=support, precisao=tp / predicted if predicted else 0,
                         recall=tp / support if support else 0,
                         f1=2 * tp / (support + predicted) if support + predicted else 0))
    hits = sum(counts[label, label] for label in labels)
    invalid = sum(v for (g, p), v in counts.items() if p == "invalid")
    return dict(n=n, acuracia=hits/n, cobertura=(n-invalid)/n, invalidos=invalid,
                macro_f1=sum(r["f1"] for r in rows)/len(rows)), rows, counts
