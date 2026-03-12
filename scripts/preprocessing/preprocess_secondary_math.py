"""this normalzies and extracts final answers from secondary math raw JSONL"""
import json
import os
import re
from typing import Dict


LATEX_NORMALIZE_REPLACE = {
    "\\left": "",
    "\\right": "",
    "∶": ":",
    "，": ",",
    "$": "",
    "\\approx": "=",
    "\\simeq": "=",
    "\\sim": "=",
    "^\\prime": "'",
    "^{\\prime}": "'",
    "^\\circ": "",
    "%": "",
}


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _normalize_answer(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.strip()
    for token, replacement in LATEX_NORMALIZE_REPLACE.items():
        s = s.replace(token, replacement)
    s = s.strip("\n$,.:;^_=+`!@#$%^&*~，。")
    s = re.sub(r"\\(?:mathrm|mathbf)\{~?([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_final_from_gsm8k(solution: str) -> str:
    if not solution:
        return ""
    nums = re.findall(r"-?\d+\.?\d*", solution)
    return nums[-1] if nums else ""


def main() -> None:
    data_dir = os.environ.get("DATA_DIR") or os.path.join(_project_root(), "data")
    in_path = os.path.join(data_dir, "secondary_math_raw.jsonl")
    out_path = os.path.join(data_dir, "secondary_math_preprocessed.jsonl")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    written = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec: Dict = json.loads(line)
            ds = rec.get("dataset", "")
            q = (rec.get("question") or "").strip()
            sol = (rec.get("solution") or "").strip()
            ans = rec.get("final_answer")

            if ds == "gsm8k" and (not ans or not ans.strip()):
                ans = _extract_final_from_gsm8k(sol)

            ans_norm = _normalize_answer(ans or "")
            row = {
                "dataset": ds,
                "question": q,
                "solution": sol,
                "final_answer": ans_norm,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} records to {out_path}")


if __name__ == "__main__":
    main()

