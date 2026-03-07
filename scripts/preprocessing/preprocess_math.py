import json
import re

# here we preprocess the data from the OlympiadBench dataset and extract questions and answers
# for the tokenizer to parse the question and answers correctly, we must normalize the latex

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
   # this normalizes the latex answer string 
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.strip()
    # strip displays math delimiters
    for token, replacement in LATEX_NORMALIZE_REPLACE.items():
        s = s.replace(token, replacement)
    # we replace special symbols here and removing \\mathrm/\\mathbf wrappers
    s = s.strip("\n$,.:;^_=+`!@#$%^&*~，。")
    s = re.sub(r"\\(?:mathrm|mathbf)\{~?([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def final_answer_to_string(final_answer) -> str:
    # we turn the final_answer (list or str) into a single normalized string
    if isinstance(final_answer, list):
        parts = [normalize_latex_answer(str(p)) for p in final_answer if p]
        return ",".join(parts) if parts else ""
    return normalize_latex_answer(str(final_answer))


def main():
    import os
    data_dir = os.environ.get("DATA_DIR") or os.path.join(os.getcwd(), "data")
    in_path = os.environ.get("MATH_OLYMPIAD_JSON", os.path.join(data_dir, "math_olympiad_questions.json"))
    out_path = os.environ.get("MATH_PREPROCESSED_JSONL", os.path.join(data_dir, "olympiad_preprocessed.jsonl"))
    # here we set the paths for the input and output files

    compilation_of_all_questions = {}
    with open(in_path, "r", encoding="utf-8") as f:
        list_of_questions = json.load(f)
        # load the list of questions from the json file

    # create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # write the questions and answers to the output file
    with open(out_path, "w", encoding="utf-8") as out:
        for item in list_of_questions:
            # get the question and solution from the item
            question = (item.get("question") or "").strip()
            solution = item.get("solution")
            if isinstance(solution, list):
                solution = "\n\n".join(s for s in solution if s)
            else:
                solution = (solution or "").strip()
            answer = final_answer_to_string(item.get("final_answer"))
            # add the question and answer to the compilation of all questions
            compilation_of_all_questions[question] = answer
            # create the row to write to the output file, this includes the question, solution, final answer, subfield, and id
            row = {"question": question, "solution": solution, "final_answer": answer, "subfield": item.get("subfield") or "", "id": item.get("id")}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(compilation_of_all_questions)} records to {out_path}")
    # return the compilation of all questions for future use and ensure that all questions are written to the output path
    return compilation_of_all_questions

if __name__ == "__main__":
    main()