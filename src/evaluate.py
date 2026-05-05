import json
import time
import os
from datetime import datetime
from tqdm import tqdm
from models import query_model, MODELS


####### Config #######
BENCHMARK_PATH = "data/benchmark.json"
RESULTS_DIR = "results/raw"
os.makedirs(RESULTS_DIR, exist_ok=True)

# List of models to evaluate.
MODELS_TO_RUN = ["llama3.2", "mistral", "phi3", "gemini-1.5-flash"]

SYSTEM_PROMPT = """You are a knowledgeable assistant answering questions about
AI, machine learning, and materials science. Answer accurately and concisely.
For questions where you are genuinely uncertain, say so clearly rather than
guessing. For calibration questions that have no single right answer, express
appropriate uncertainty."""

####### Load benchmark #######
with open(BENCHMARK_PATH) as f:
    benchmark = json.load(f)

print(f"Loaded {len(benchmark)} questions")
print(f"Question types: {set(q['type'] for q in benchmark)}")

def build_prompt(question_item):
    prompt = question_item["question"]
    if question_item.get("context"):
        prompt = f"Context:\n{question_item['context']}\n\nQuestion: {prompt}"
    return prompt


####### Run evaluation for each model #######
for model_key in MODELS_TO_RUN:
    print(f"\n{'='*50}")
    print(f"Evaluating: {MODELS[model_key]['display_name']}")
    print(f"{'='*50}\n")

    results = {.
        "model": model_key,
        "model_info": MODELS[model_key],
        "timestamp": datetime.now().isoformat(),
        "responses": []
    }

    for i, item in enumerate(tqdm(benchmark, desc=model_key)):
        prompt = build_prompt(item)

        start_time = time.time()
        response = query_model(model_key, prompt, system_prompt=SYSTEM_PROMPT)
        latency = time.time() - start_time

        results["responses"].append({
            "id": item["id"],
            "type": item["type"],
            "domain": item["domain"],
            "difficulty": item["difficulty"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "rubric": item["rubric"],
            "model_response": response,
            "latency_seconds": latency,
            "score": None,           # filled in by judge.py
            "score_reasoning": None  # filled in by judge.py
        })

        # Print progress every 10 questions
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{len(benchmark)}")

    # Save raw results for this model
    output_path = f"{RESULTS_DIR}/{model_key}_responses.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved responses to {output_path}")

print("\nEvaluation complete for all models.")