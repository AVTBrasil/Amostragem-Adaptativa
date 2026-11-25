import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from avaliacao.__main__ import run
from avaliacao.experimentos import classification, gsm, mt, strategies
from avaliacao.judge import QwenJudge, parse


class EvaluationTests(unittest.TestCase):
    def test_qwen_backend(self):
        tokenizer, torch, hf = Mock(), Mock(), Mock()
        hf.AutoTokenizer.from_pretrained.return_value = tokenizer
        class Batch(dict):
            input_ids = Mock(shape=(1, 7))

        tokenizer.return_value.to.return_value = Batch()
        from contextlib import nullcontext
        torch.inference_mode.return_value = nullcontext()
        model = hf.AutoModelForCausalLM.from_pretrained.return_value.eval.return_value
        model.config.max_position_embeddings = 100
        model.generate.return_value = [[0]*7 + [42]]
        tokenizer.decode.return_value = '{"verdict":"correct"}'
        with patch.dict("sys.modules", torch=torch, transformers=hf):
            self.assertEqual(QwenJudge()([]), '{"verdict":"correct"}')
        tokenizer.decode.assert_called_once_with([42], skip_special_tokens=True)
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_sampling(self):
        result = strategies([4]*16, 4, "")
        self.assertEqual(result["adaptativa 0.90"]["amostras"], 3)
        self.assertEqual(result["adaptativa 0.95"]["amostras"], 4)
        self.assertEqual(strategies([None]*8, 4, "")["adaptativa 0.95"]["resposta"], None)
        self.assertEqual(strategies([2, 1, 1, 2], 2, "")["fixa 4"]["resposta"], 2)
        row = dict(gabarito=4, pergunta="q", respostas=[dict(texto="x", numero=v) for v in [2,2,2,4]])
        result = gsm(row, Mock(side_effect=["incorrect"]*3 + ["correct"]))["strategies"]
        self.assertFalse(result["fixa 4"]["acerto"])
        self.assertTrue(result["fixa 4 + juiz"]["acerto"])

    def test_invalid(self):
        self.assertEqual(parse("gsm8k", "correct or incorrect"), "invalid")
        metrics, details, counts = classification("gsm8k", [("correct", "invalid"), ("incorrect", "incorrect")])
        self.assertEqual(metrics["acuracia"], 0.5)
        self.assertEqual(metrics["cobertura"], 0.5)
        self.assertEqual(counts["correct", "invalid"], 1)

    def test_mt(self):
        history = [dict(role=r, content=str(i)) for i, r in enumerate(["user", "assistant"]*2)]
        row = dict(turn=1, winner="model_b", conversation_a=history, conversation_b=history)
        ask = Mock(return_value="B")
        self.assertEqual(mt(row, ask)["pairs"], [("B", "B")])
        self.assertEqual(ask.call_args.args[0], {"A": history[:2], "B": history[:2]})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run("mtbench", [row, dict(row, turn=2)], output, "fixture", Mock(return_value='{"verdict":"B"}'))
            with (output/"metricas.csv").open(encoding="utf-8-sig") as f:
                self.assertEqual([r["grupo"] for r in csv.DictReader(f)], ["total", "turno 1", "turno 2"])

    def test_incremental_resume(self):
        row = dict(indice=10, gabarito=8, pergunta="q",
                   respostas=[dict(texto=str(i), numero=i) for i in range(4)])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = ("gsm8k", [row], out, "fixture")
            fail = Mock(side_effect=['{"verdict":"incorrect"}', RuntimeError("interrupção")])
            with self.assertRaises(RuntimeError):
                run(*args, fail)
            judge = Mock(return_value='{"verdict":"incorrect"}')
            run(*args, judge)
            self.assertEqual(judge.call_count, 3)
            run(*args, Mock(side_effect=AssertionError))
            self.assertTrue((out/"estrategias.png").stat().st_size > 1000)
            with (out/"metricas.csv").open(encoding="utf-8-sig") as f:
                self.assertEqual(next(csv.DictReader(f))["n"], "4")
            with self.assertRaises(ValueError):
                run("gsm8k", [row], out, "changed", judge)
