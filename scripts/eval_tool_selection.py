"""Точность выбора инструмента на 100 сценариях: правила против LLM.

    python scripts/eval_tool_selection.py --methods rules          # без ключей, мгновенно
    python scripts/eval_tool_selection.py --methods rules,llm      # нужен LLM_PROVIDER, ключ и модель в .env (или --model)
    python scripts/eval_tool_selection.py --methods llm --delay 2  # пауза между запросами против лимитов API

Метрики и их смысл — в agent_module/routing/evaluate.py. Результаты: results/tool_selection_<метод>.json/.md,
подробный разбор по сценариям — в поле ``rows`` JSON.
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Тихая консоль: по умолчанию только предупреждения и ошибки; подробности — в results/eval_tool_selection.log.
# Значения из окружения и аргументов командной строки имеют приоритет над этими.
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOG_TO_FILE", "0")
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

MAX_CONSECUTIVE_ERRORS = 5
LOG_PATH = ROOT / "results" / "eval_tool_selection.log"

from agent_module.routing.evaluate import (  # noqa: E402
    DEFAULT_SCENARIOS_PATH,
    load_scenarios,
    make_llm_predictor,
    predict_rules,
    run,
    summarize,
)


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}" if isinstance(value, float) and value <= 1 else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", default="rules,llm", help="через запятую: rules, llm")
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS_PATH))
    parser.add_argument("--limit", type=int, default=0, help="взять только первые N сценариев")
    parser.add_argument("--delay", type=float, default=0.0, help="пауза между запросами к LLM, сек")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--model", default="", help="модель вместо заданной в .env; можно несколько через запятую — каждая получит свой прогон и файл")
    parser.add_argument("--prompt", choices=["full", "minimal"], default="full", help="minimal — абляция: без таблицы правил в промпте, только описания инструментов")
    parser.add_argument("--mask-step-tools", action="store_true", help="абляция: убрать названия инструментов из шага плана")
    parser.add_argument("--allow-fallbacks", action="store_true", help="не отключать запасные модели (по умолчанию в оценке они выключены)")
    parser.add_argument("--show-errors", action="store_true", help="печатать ошибочные сценарии")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
    if args.limit:
        scenarios = scenarios[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global LOG_PATH
    LOG_PATH = out_dir / "eval_tool_selection.log"
    eval_logger = logging.getLogger("ege_ai_agent.eval")
    eval_logger.setLevel(logging.DEBUG)
    eval_logger.propagate = False
    handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    eval_logger.addHandler(handler)

    def save(stem: str, label: str, rows: list) -> None:
        (out_dir / f"{stem}.json").write_text(
            json.dumps({"method": label, "summary": summarize(rows), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def progress(stem: str, label: str, total: int):
        started = time.perf_counter()
        streak = {"errors": 0}

        def on_row(row: dict, rows: list) -> None:
            done = len(rows)
            mark = "ok " if row["correct"] else "ERR" if row["error"] else "-- "
            print(f"[{done:>3}/{total}] {mark} {row['id']}  ожидали {row['expected']:<22} получили {row['predicted']}", flush=True)
            if row["error"]:
                print(f"          причина: {row['error']}", flush=True)
                streak["errors"] += 1
            else:
                streak["errors"] = 0
            if done % 10 == 0:
                save(stem, label, rows)  # промежуточное сохранение: прогон можно прервать
                ok = sum(r["correct"] for r in rows)
                elapsed = time.perf_counter() - started
                eta = elapsed / done * (total - done)
                print(f"          --- {done}/{total}: верно {ok}, ошибок API {sum(1 for r in rows if r['error'])}, прошло {elapsed:.0f} с, осталось ~{eta:.0f} с", flush=True)
            if streak["errors"] >= MAX_CONSECUTIVE_ERRORS:
                save(stem, label, rows)
                raise SystemExit(
                    f"\nПрерываю: {MAX_CONSECUTIVE_ERRORS} ошибок API подряд (последняя: {row['error']}).\n"
                    f"Проверьте ключ, название модели и квоту. Полный трейсбек: {LOG_PATH}"
                )

        return on_row

    variant = ("_minimal" if args.prompt == "minimal" else "") + ("_masked" if args.mask_step_tools else "")
    suffix = variant + ("" if Path(args.scenarios).resolve() == Path(DEFAULT_SCENARIOS_PATH).resolve() else "_" + Path(args.scenarios).stem)
    jobs = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        if method == "rules":
            jobs.append((method, None))
        elif method == "llm":
            models = [m.strip() for m in args.model.split(",") if m.strip()] or [None]
            jobs.extend((method, model) for model in models)
        else:
            raise SystemExit(f"Неизвестный метод: {method} (rules | llm)")

    summaries = {}
    for method, model in jobs:
        if method == "rules":
            label, stem = "rules", "tool_selection_rules" + suffix[len(variant):]
            rows = run(scenarios, predict_rules)
        else:
            if not args.allow_fallbacks:
                os.environ.setdefault("LLM_FALLBACKS", "0")  # измеряем одну модель, а не цепочку запасных
            from agent_module.llms import build_main_llm, provider, resolve_model_name

            llm = build_main_llm(model)
            name = resolve_model_name(provider(), model)
            label = f"llm ({provider()}/{name}" + (", промпт minimal" if args.prompt == "minimal" else "") + (", шаг без названий инструментов" if args.mask_step_tools else "") + ")"
            stem = "tool_selection_llm_" + "".join(c if c.isalnum() or c in ".-" else "_" for c in name) + suffix
            print(f"\n=== {label} ===", flush=True)
            rows = run(
                scenarios,
                make_llm_predictor(llm, prompt=args.prompt, mask_step=args.mask_step_tools),
                pause=lambda: time.sleep(args.delay) if args.delay else None,
                on_row=progress(stem, label, len(scenarios)),
            )
        summary = summarize(rows)
        summaries[label] = summary
        save(stem, label, rows)
        kinds = Counter((r["error"] or "").split(":")[0] for r in rows if r["error"])
        if kinds:
            print(f"\nОшибки API по типам ({label}): " + ", ".join(f"{k} ×{v}" for k, v in kinds.most_common()))
        if args.show_errors:
            print(f"\nОшибки ({label}):")
            for r in rows:
                if not r["correct"]:
                    print(f"  {r['id']}: «{r['user_text'][:70]}» ожидали {r['expected']}, получили {r['predicted']}" + (f" [{r['error']}]" if r["error"] else ""))

    header = "| Метод | Tool Selection Accuracy | Когда нужен инструмент | Когда не нужен | Ложные вызовы | Ошибки API | p50, мс |"
    lines = [header, "|---|---|---|---|---|---|---|"]
    for label, s in summaries.items():
        lines.append(
            f"| {label} | {_fmt(s['tool_selection_accuracy'])} | {_fmt(s['accuracy_when_tool_needed'])} | "
            f"{_fmt(s['accuracy_when_no_tool_needed'])} | {s['false_tool_calls']} | {s['errors']} | {_fmt(s['latency_ms_p50'])} |"
        )
    table = "\n".join(lines)
    print(f"Сценариев: {len(scenarios)}\n{table}")
    (out_dir / f"tool_selection{suffix}.md").write_text(f"# Выбор инструмента: правила против LLM\n\nСценариев: {len(scenarios)}\n\n{table}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
