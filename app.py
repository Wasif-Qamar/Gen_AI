import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None
if OPENAI_API_KEY and OpenAI:
    try:
        print(f"Initializing OpenAI client with API key: {OPENAI_API_KEY[:10]}...")
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully")
    except Exception as e:
        print(f"Error initializing OpenAI client: {str(e)}")
        client = None
else:
    print("OpenAI API key or client not available")


def build_prompt(code: str, language: str) -> str:
    return (
        "You are an expert code reviewer and fixer. "
        "Given the user's source code and language, do the following strictly and ONLY return a valid JSON object: "
        "1) analysis: A thorough list of syntax issues, logical bugs, edge cases, and improvements. "
        "2) fixed_code: The corrected full code with clear inline comments explaining changes. "
        "3) documentation: A detailed explanation of what was wrong, what was changed, and how the corrected code works. "
        "Important: Return JSON with keys 'analysis', 'fixed_code', and 'documentation' only. No markdown, no code fences. "
        f"Language: {language}.\n\nOriginal Code:\n{code}"
    )


def call_llm(code: str, language: str):
    if not client:
        # Fallback mock response if API key is missing or client failed to initialize
        analysis = (
            "OpenAI API key not configured or client unavailable. "
            "Returning a mock review so you can test the app plumbing."
        )
        fixed_code = (
            f"// Mock fixed code for {language}.\n"
            "// Add your real OPENAI_API_KEY in the backend .env to enable real reviews.\n"
            f"{code}"
        )
        documentation = (
            "This is a mock documentation block. Once the backend is configured with your "
            "OpenAI API key, the app will produce real analysis and corrections."
        )
        return {"analysis": analysis, "fixed_code": fixed_code, "documentation": documentation}

    prompt = build_prompt(code, language)

    try:
        # Prefer Responses API if available, otherwise fall back to chat.completions
        try:
            resp = client.responses.create(
                model=MODEL_NAME,
                input=prompt,
                temperature=0.2,
            )
            text = resp.output_text
        except Exception:
            chat = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": "You are a strict code-review JSON generator."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
            )
            text = chat.choices[0].message.content

        # Ensure we parse valid JSON out of the model output
        # Some models may wrap the JSON in code fences; try to extract JSON substring
        def extract_json(s: str):
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1 and end > start:
                return s[start:end+1]
            return s

        json_str = extract_json(text)
        data = json.loads(json_str)

        # Basic validation
        for key in ["analysis", "fixed_code", "documentation"]:
            if key not in data:
                data[key] = ""
        return data
    except Exception as e:
        return {
            "analysis": f"LLM error: {str(e)}",
            "fixed_code": code,
            "documentation": "Failed to produce corrected output due to API error.",
        }


@app.route('/review_code', methods=['POST'])
def review_code():
    try:
        payload = request.get_json(force=True)
        code = payload.get('code', '')
        language = payload.get('language', 'Plaintext')

        if not code.strip():
            return jsonify({
                "error": "No code provided",
            }), 400

        result = call_llm(code, language)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', '5001'))
    app.run(host='0.0.0.0', port=port, debug=True)