import json
import re

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


def normalize_latex_answer(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.strip()
    for token, replacement in LATEX_NORMALIZE_REPLACE.items():
        s = s.replace(token, replacement)
    s = s.strip("\n$,.:;^_=+`!@#$%^&*~，。")
    s = re.sub(r"\\(?:mathrm|mathbf)\{~?([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def final_answer_to_string(final_answer) -> str:
    if isinstance(final_answer, list):
        parts = [normalize_latex_answer(str(p)) for p in final_answer if p]
        return ",".join(parts) if parts else ""
    return normalize_latex_answer(str(final_answer))


def main():
    import os
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
    data_dir = os.environ.get("DATA_DIR") or os.path.join(_project_root, "data")
    in_path = os.environ.get("MATH_OLYMPIAD_JSON", os.path.join(data_dir, "math_olympiad_questions.json"))
    out_path = os.environ.get("MATH_PREPROCESSED_JSONL", os.path.join(data_dir, "olympiad_preprocessed.jsonl"))

    compilation_of_all_questions = {}
    with open(in_path, "r", encoding="utf-8") as f:
        list_of_questions = json.load(f)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for item in list_of_questions:
            question = (item.get("question") or "").strip()
            solution = item.get("solution")
            if isinstance(solution, list):
                solution = "\n\n".join(s for s in solution if s)
            else:
                solution = (solution or "").strip()
            answer = final_answer_to_string(item.get("final_answer"))
            compilation_of_all_questions[question] = answer
            row = {"question": question, "solution": solution, "final_answer": answer, "subfield": item.get("subfield") or "", "id": item.get("id")}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(compilation_of_all_questions)} records to {out_path}")
    return compilation_of_all_questions

if __name__ == "__main__":
    main()