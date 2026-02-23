import json

#note: this data is from olympiadbench and we extract the questions and the answers 

def main():
    compilation_of_all_questions = {}
    with open("data/math_olympiad_questions.json", "r") as f:
        list_of_questions = json.load(f)
        for item in list_of_questions:
            question = item["question"]
            answer = item["final_answer"]
            compilation_of_all_questions[question] = answer
    return compilation_of_all_questions

if __name__ == "__main__":
    result = main()
    print(result)