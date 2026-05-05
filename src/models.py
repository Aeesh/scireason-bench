import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import ollama

load_dotenv()

MODELS = {
    "llama3.2": {
        "type": "ollama",
        "display_name": "Llama 3.2 3B",
        "provider": "Meta (local)",
        "params": "3B"
    },
    "mistral": {
        "type": "ollama",
        "display_name": "Mistral 7B",
        "provider": "Mistral AI (local)",
        "params": "7B"
    },
    "phi3": {
        "type": "ollama",
        "display_name": "Phi-3 Mini",
        "provider": "Microsoft (local)",
        "params": "3.8B"
    },
    "gemini-1.5-flash": {
        "type": "gemini",
        "display_name": "Gemini 1.5 Flash",
        "provider": "Google (cloud)",
        "params": "unknown"
    }
}

# Function to query Ollama models with optional system prompt and return the response content as a string.
# The system prompt is included as a message with role "system" if provided.
def query_ollama(model_name, prompt, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

        response = ollama.chat(model=model_name, messages=messages)
        return response["message"]["content"]


# Function to query Gemini models with optional system prompt.
# Configures the API key, creates a generative model instance, and generates content based on the prompt.
# Returns the response text.
def query_gemini(prompt, system_prompt=None):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction=system_prompt or ""
    )
    response = model.generate_content(prompt)
    return response.text


# Main function to query a model based on the provided model key, prompt, and optional system prompt.
# It looks up the model configuration, determines the type (Ollama or Gemini), and calls the appropriate query function.
# Implements retry logic with exponential backoff in case of failures, and returns the response.
def query_model(model_key, prompt, system_prompt=None, retries=3):
    config = MODELS[model_key]
    for attempt in range(retries):
        try:
            if config["type"] == "ollama":
                return query_ollama(model_key, prompt, system_prompt)
            elif config["type"] == "gemini":
                time.sleep(1)  # respect rate limits
                return query_gemini(prompt, system_prompt)
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed after {retries} attempts: {e}")
                return f"ERROR: {str(e)}"
            time.sleep(2 ** attempt)  # exponential backoff
    return "ERROR: max retries exceeded"

