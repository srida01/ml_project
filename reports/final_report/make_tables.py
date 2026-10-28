#!/usr/bin/env python3
"""Generate LaTeX macros and tables for the report from saved experiment outputs.

Run from the project root after both experiments and their analyses finished:
    python reports/final_report/make_tables.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "generated"
TABLES = ROOT / "reports" / "tables"
E1, E2 = "exp1_row_split", "exp2_file_split"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def tex(text: object) -> str:
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}"


def num(value: float, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def short_file(name: str) -> str:
    """Readable capture name, e.g. 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv' -> 'Fri PM (DDoS)'."""
    stem = name.replace(".pcap_ISCX.csv", "")
    day = stem.split("-")[0][:3]
    part = " AM" if "Morning" in stem else " PM" if "Afternoon" in stem else ""
    tag = stem.split("-")[-1] if part else ""
    tag = {"DDos": "DDoS", "Infilteration": "Infiltration"}.get(tag, tag)
    return f"{day}{part}" + (f" ({tag})" if tag and tag != "Morning" else "")


def runtime(log: Path) -> tuple[str, str]:
    text = log.read_text()
    elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", text).group(1)
    peak_kb = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", text).group(1))
    minutes, seconds = elapsed.split(":")[-2:]
    return f"{int(minutes)} min {round(float(seconds))} s", f"{peak_kb / 1024 / 1024:.1f}"


def table(caption: str, label: str, header: list[str], rows: list[list[str]], spec: str, note: str = "", fit: bool = False) -> str:
    lines = [r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
             (r"\resizebox{\linewidth}{!}{" if fit else "") + rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}", r"\toprule", " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}" + ("}" if fit else "")]
    if note:
        lines.append(rf"\par\smallskip\parbox{{\linewidth}}{{\footnotesize {note}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    a1, a2 = load(TABLES / E1 / "analysis.json"), load(TABLES / E2 / "analysis.json")
    s1, s2 = a1["system_metrics"], a2["system_metrics"]
    overlap1, overlap2 = load(TABLES / E1 / "split_overlap.json"), load(TABLES / E2 / "split_overlap.json")
    round1, round2 = load(TABLES / "svm_tuning_round1.json"), load(TABLES / "svm_tuning_round2.json")
    untuned = load(TABLES / "exp1_untuned_test_system_metrics.json")
    download = load(ROOT / "data" / "manifests" / "cicids2017_download.json")
    quality = load(ROOT / "data" / "manifests" / "data_quality_report.json")
    run1, mem1 = runtime(ROOT / "reports" / "experiments" / f"{E1}.log")

    total = a1["total_rows"]
    infinite = sum(sum(f["infinite_values_replaced"].values()) for f in quality["files"])
    duplicates = sum(f["exact_duplicate_rows"] for f in quality["files"])
    families_total = {}
    for split in a1["split_family_counts"].values():
        for family, count in split.items():
            families_total[family] = families_total.get(family, 0) + count
    before = next(r for r in round1["results"] if r["approx_components"] == 512 and r["alpha"] == 1e-4 and r["epochs"] == 1)
    selected = round2["selected"]
    gamma = selected["gamma"]
    stack1 = next(r for r in a1["multiclass_comparison"] if r["model"].startswith("Stacked"))
    groups2 = a2["split_manifest"]["groups"]

    macros = {
        "TotalRows": f"{total:,}", "NumFeatures": str(a1["feature_count"]), "BenignPct": pct(families_total["BENIGN"] / total, 1),
        "DupRows": f"{duplicates:,}", "DupPct": pct(duplicates / total, 1), "InfCount": f"{infinite:,}",
        "EoneTrain": f"{a1['split_manifest']['rows']['train']:,}", "EoneVal": f"{a1['split_manifest']['rows']['validation']:,}",
        "EoneTest": f"{a1['split_manifest']['rows']['test']:,}",
        "EtwoTrainFiles": ", ".join(tex(short_file(f)) for f in groups2["train"]),
        "EtwoValFiles": ", ".join(tex(short_file(f)) for f in groups2["validation"]),
        "EtwoTestFiles": ", ".join(tex(short_file(f)) for f in groups2["test"]),
        "EtwoTestRows": f"{a2['split_manifest']['rows']['test']:,}",
        "EoneFusionRecall": pct(s1["fusion"]["recall"]), "EoneFusionFpr": pct(s1["fusion"]["fpr"]),
        "EoneFusionPrecision": pct(s1["fusion"]["precision"]), "EoneFusionAlerts": f"{s1['fusion']['alert_volume_per_1000_benign']:.1f}",
        "EoneGateRecall": pct(s1["svm_gate"]["recall"]), "EoneGateFpr": pct(s1["svm_gate"]["fpr"]),
        "EoneStackMacroFone": num(stack1["macro_f1"]), "EoneStackAccuracy": pct(stack1["accuracy"]),
        "EoneCascadeMacroFone": num(s1["known_attack_cascade"]["macro_f1"]),
        "EoneMsPerFlow": f"{s1['inference_ms_per_flow']:.3f}", "EoneFlowsPerSec": f"{s1['inference_flows_per_second']:,.0f}",
        "EoneAeAuc": num(a1["autoencoder"]["roc_auc"]), "EoneAeTpr": pct(a1["autoencoder"]["attack_tpr"], 1),
        "EoneAeFpr": pct(a1["autoencoder"]["benign_fpr"]),
        "EtwoGateRecall": pct(s2["svm_gate"]["recall"]), "EtwoGateFpr": pct(s2["svm_gate"]["fpr"]),
        "EtwoFusionRecall": pct(s2["fusion"]["recall"]), "EtwoFusionFpr": pct(s2["fusion"]["fpr"]),
        "EtwoFusionAlerts": f"{s2['fusion']['alert_volume_per_1000_benign']:.1f}",
        "EtwoAeTpr": pct(a2["autoencoder"]["attack_tpr"], 1), "EtwoAeFpr": pct(a2["autoencoder"]["benign_fpr"]),
        "EtwoUnseen": ", ".join(a2["unseen_test_families"]) or "none",
        "EoneOverlap": pct(overlap1["test_fraction_with_exact_copy_in_train"], 1),
        "EtwoOverlap": pct(overlap2["test_fraction_with_exact_copy_in_train"], 1),
        "SvmValFprBefore": pct(before["validation_fpr"], 1), "SvmValFprAfter": pct(selected["validation_fpr"], 1),
        "SvmSelected": (rf"$D={selected['approx_components']}$ random features, $\gamma={gamma}$, $\alpha=10^{{{round(__import__('math').log10(selected['alpha']))}}}$, "
                        rf"{selected['epochs']} SGD epochs, {selected['max_exact_rows']:,} training rows"),
        "UntunedGateFpr": pct(untuned["svm_gate"]["fpr"]), "UntunedFusionAlerts": f"{untuned['fusion']['alert_volume_per_1000_benign']:.1f}",
        "BaselineRows": f"{a1['baseline_training_rows']:,}", "EoneRuntime": run1, "EonePeakMem": mem1,
        "ArchiveShaShort": download["archive_sha256"][:16],
    }
    model_keys = {"Logistic regression": "Lr", "Random forest": "Rf", "XGBoost": "Xgb", "SVM gate (validation-tuned)": "Gate"}
    for prefix, analysis in (("Eone", a1), ("Etwo", a2)):
        for row in analysis["binary_comparison"]:
            key = model_keys.get(row["model"])
            if key is None:
                continue
            # setdefault keeps the earlier system-level definitions (e.g. gate FPR with 2 decimals).
            macros.setdefault(f"{prefix}{key}Recall", pct(row["recall"]))
            macros.setdefault(f"{prefix}{key}Precision", pct(row["precision"]))
            macros.setdefault(f"{prefix}{key}Fpr", pct(row["fpr"], 3))
            macros.setdefault(f"{prefix}{key}Alerts", f"{row['alerts_per_1000_benign']:.1f}")
            macros.setdefault(f"{prefix}{key}Auc", num(row["roc_auc"], 4))
        for row in analysis["multiclass_comparison"]:
            key = {"Random forest": "Rf", "XGBoost": "Xgb", "Logistic regression": "Lr"}.get(row["model"], "Stack")
            macros[f"{prefix}{key}MacroFone"] = num(row["macro_f1"])
    macros["EtwoAeAuc"] = num(a2["autoencoder"]["roc_auc"])
    (OUT / "macros.tex").write_text("".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in macros.items()))

    # Files and families
    rows = []
    for name, info in sorted(a1["per_file"].items(), key=lambda kv: kv[0]):
        attacks = ", ".join(f"{fam} ({cnt:,})" for fam, cnt in sorted(info["families"].items(), key=lambda kv: -kv[1]) if fam != "BENIGN")
        rows.append([tex(short_file(name)), f"{info['rows']:,}", f"{info['families'].get('BENIGN', 0):,}", attacks or "none"])
    (OUT / "table_files.tex").write_text(
        table("CICIDS2017 capture files and their traffic", "tab:files", ["File", "Flows", "Benign", "Attack families (flows)"], rows, "lrrp{6.2cm}")
        .replace(r"\small", r"\footnotesize"))

    # SVM tuning
    candidates = sorted(round2["results"], key=lambda r: r["validation_fpr"])[:6]
    rows = [[str(before["approx_components"]), r"$10^{-4}$", "1", "scale", "100k", pct(before["validation_fpr"]), num(before["validation_roc_auc"], 4), f"{before['fit_seconds']:.1f}"]]
    for r in candidates:
        mark = (lambda s: rf"\textbf{{{s}}}") if r is candidates[0] else (lambda s: s)
        alpha = rf"$10^{{{round(__import__('math').log10(r['alpha']))}}}$"
        rows.append([mark(str(r["approx_components"])), alpha, mark(str(r["epochs"])), mark(str(r["gamma"])), mark(f"{r['max_exact_rows'] // 1000}k"),
                     mark(pct(r["validation_fpr"])), mark(num(r["validation_roc_auc"], 4)), mark(f"{r['fit_seconds']:.1f}")])
    (OUT / "table_svm_tuning.tex").write_text(table(
        "SVM gate hyperparameter search on the validation split (first row: original setting; bold: selected)", "tab:svmtuning",
        ["$D$", r"$\alpha$", "Epochs", r"$\gamma$", "Rows", r"Val.\ FPR (\%)", r"Val.\ AUC", "Fit (s)"], rows, "rrrrrrrr",
        f"FPR measured at the threshold giving 98\\% validation recall. {len(round1['results']) + len(round2['results'])} configurations were evaluated in two rounds; the six best of round 2 are shown."))

    # Binary comparison
    rows = []
    for r in a1["binary_comparison"]:
        rows.append([tex(r["model"]), pct(r["precision"]), pct(r["recall"]), pct(r["f1"]), pct(r["fpr"], 3), f"{r['alerts_per_1000_benign']:.1f}",
                     num(r["roc_auc"], 4), f"{r['ms_per_flow']:.4f}"])
    (OUT / "table_binary.tex").write_text(table(
        "Binary detection on the Experiment 1 test set (attack = positive class)", "tab:binary",
        ["Model", "Prec.\\,\\%", "Rec.\\,\\%", "F1\\,\\%", "FPR\\,\\%", "Alerts/1k", "ROC-AUC", "ms/flow"], rows, "lrrrrrrr", fit=True, note="Baselines use a 0.5 probability threshold; the SVM gate uses its validation-tuned threshold. ms/flow for baselines and the gate is model scoring only; for the fused system it is the complete engine including validation and fusion."))

    rows = [[tex(r["model"]), pct(r["accuracy"]), num(r["macro_f1"]), num(r["weighted_f1"]), pct(r["balanced_accuracy"])] for r in a1["multiclass_comparison"]]
    (OUT / "table_multiclass.tex").write_text(table(
        "Attack-family attribution on true attack flows of the Experiment 1 test set", "tab:multiclass",
        ["Model", "Accuracy\\,\\%", "Macro-F1", "Weighted-F1", "Bal.\\ acc.\\,\\%"], rows, "lrrrr", fit=True, note=
        f"Base models are the stack's own base learners refitted on its {a1['stack_classes'].__len__()}-family training sample. The end-to-end cascade (gate misses counted as errors) has macro-F1 {num(s1['known_attack_cascade']['macro_f1'])}."))

    per_class = a1["stack_per_class"]
    rows = [[tex(f), pct(v["precision"]), pct(v["recall"]), pct(v["f1"]), f"{v['support']:,}"] for f, v in sorted(per_class.items(), key=lambda kv: -kv[1]["support"]) if v["support"]]
    (OUT / "table_perclass.tex").write_text(table("Per-family results of the stacked ensemble (Experiment 1 test set)", "tab:perclass",
                                                  ["Family", "Precision\\,\\%", "Recall\\,\\%", "F1\\,\\%", "Test flows"], rows, "lrrrr"))

    def family_list(a):
        return ", ".join(sorted(f for f in a["split_family_counts"]["test"] if f != "BENIGN"))

    stack2 = next(r for r in a2["multiclass_comparison"] if r["model"].startswith("Stacked"))
    rows = [
        ["Test flows", f"{a1['split_manifest']['rows']['test']:,}", f"{a2['split_manifest']['rows']['test']:,}"],
        ["Attack families in test", tex(family_list(a1)), tex(family_list(a2))],
        ["Families unseen in training", tex(", ".join(a1["unseen_test_families"]) or "none"), tex(", ".join(a2["unseen_test_families"]) or "none")],
        ["Test flows with exact copy in train", pct(overlap1["test_fraction_with_exact_copy_in_train"], 1) + r"\%", pct(overlap2["test_fraction_with_exact_copy_in_train"], 1) + r"\%"],
        ["SVM gate recall / FPR", f"{pct(s1['svm_gate']['recall'])} / {pct(s1['svm_gate']['fpr'])}", f"{pct(s2['svm_gate']['recall'])} / {pct(s2['svm_gate']['fpr'])}"],
        ["Autoencoder attack TPR / benign FPR", f"{pct(a1['autoencoder']['attack_tpr'])} / {pct(a1['autoencoder']['benign_fpr'])}", f"{pct(a2['autoencoder']['attack_tpr'])} / {pct(a2['autoencoder']['benign_fpr'])}"],
        ["Fused recall / FPR", f"{pct(s1['fusion']['recall'])} / {pct(s1['fusion']['fpr'])}", f"{pct(s2['fusion']['recall'])} / {pct(s2['fusion']['fpr'])}"],
        ["Fused alerts per 1000 benign", f"{s1['fusion']['alert_volume_per_1000_benign']:.1f}", f"{s2['fusion']['alert_volume_per_1000_benign']:.1f}"],
        ["Stack accuracy on attack flows", pct(stack1["accuracy"]) + r"\%", pct(stack2["accuracy"]) + r"\%"],
    ]
    (OUT / "table_experiments.tex").write_text(table("Experiment 1 (family-stratified rows) vs.\\ Experiment 2 (capture-file split)", "tab:experiments",
                                                     ["Metric", "Experiment 1", "Experiment 2"], rows, "p{4.8cm}p{3.6cm}p{3.6cm}"))

    # Appendix: detection by family in both experiments, tuning effect on test, alert levels
    rows = []
    for family in sorted(set(a1["detection_rate_by_family"]) | set(a2["detection_rate_by_family"])):
        d1, d2 = a1["detection_rate_by_family"].get(family), a2["detection_rate_by_family"].get(family)
        cell = lambda d, key: "--" if d is None or d.get(key) is None else pct(d[key], 1)
        rows.append([tex(family), cell(d1, "svm_gate"), cell(d1, "autoencoder"), cell(d2, "svm_gate"), cell(d2, "autoencoder")])
    appendix = table("Fraction of test flows flagged per family (\\%); benign row = false-positive rate", "tab:detfamily",
                     ["Family", "E1 gate", "E1 AE", "E2 gate", "E2 AE"], rows, "lrrrr")
    rows = [["SVM gate recall (\\%)", pct(untuned["svm_gate"]["recall"]), pct(s1["svm_gate"]["recall"])],
            ["SVM gate FPR (\\%)", pct(untuned["svm_gate"]["fpr"]), pct(s1["svm_gate"]["fpr"])],
            ["Fused recall (\\%)", pct(untuned["fusion"]["recall"]), pct(s1["fusion"]["recall"])],
            ["Fused alerts per 1000 benign", f"{untuned['fusion']['alert_volume_per_1000_benign']:.1f}", f"{s1['fusion']['alert_volume_per_1000_benign']:.1f}"]]
    appendix += table("Effect of validation-based SVM tuning on the Experiment 1 test set", "tab:tuningeffect", ["Metric", "Before tuning", "After tuning"], rows, "lrr")
    rows = [[tex(r["feature"]), f"{r['importance']:.4f}"] for r in a1["rf_top_features"][:10]]
    appendix += table("Ten most important features (Random Forest base model)", "tab:features", ["Feature", "Importance"], rows, "lr")
    (OUT / "table_appendix.tex").write_text(appendix)
    print("Wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
