import json
import os
import time
from tqdm import tqdm
from models import query_gemini

RESULTS_DIR = "results/raw"
SCORED_DIR = "results/scored"
os.makedirs(SCORED_DIR, exist_ok=True)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of
AI model responses to scientific and technical questions. You will be given:
- A question
- A rubric describing what a good answer must contain
- The model's response

Score the response from 0 to 3:
0 = Completely wrong, missing all key points, or confidently incorrect
1 = Partially correct, captures some key points but misses important ones
2 = Mostly correct, captures most key points with minor gaps or errors
3 = Fully correct, addresses all rubric points accurately

For calibration questions (where the model should express uncertainty):
3 = Appropriately expresses uncertainty and explains why
2 = Expresses some uncertainty but is overconfident in places
1 = Mostly confident when it should be uncertain
0 = Confidently wrong on a question with no clear answer

Respond with ONLY a JSON object:
{"score": <0-3>, "reasoning": "<1-2 sentence explanation>"}"""


def judge_response(question, rubric, model_response, question_type):
    prompt = f"""Question: {question}

Rubric (what a good answer must contain): {rubric}

Model response: {model_response}

Question type: {question_type}

Evaluate this response and return a JSON score."""

    for attempt in range(3):
        try:
            result = query_gemini(prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
            # Extract JSON from the response
            import re
            # Use regex to find the first JSON object in the response
            json_match = re.search(r'\{[^}]+\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if "score" in parsed and "reasoning" in parsed:
                    return parsed
        except Exception as e:
            print(f"Judge error on attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)  # exponential backoff

    return {"score": -1, "reasoning": "Judge failed"}


####### Score all model responses #######
model_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith("_responses.json")]

for filename in model_files:
    model_key = filename.replace("_responses.json", "")
    print(f"\nScoring: {model_key}")

    with open(f"{RESULTS_DIR}/{filename}") as f:
        results = json.load(f)

    for item in tqdm(results["responses"], desc=f"Judging {model_key}"):
        if item["score"] is not None:
            continue  # already scored, skip

        judgement = judge_response(
            question=item["question"],
            rubric=item["rubric"],
            model_response=item["model_response"],
            question_type=item["type"]
        )

        item["score"] = judgement["score"]
        item["score_reasoning"] = judgement["reasoning"]
        time.sleep(5)  # rate limit

    # Save scored results
    output_path = f"{SCORED_DIR}/{model_key}_scored.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved scored results to {output_path}")

print("\nAll scoring complete")

