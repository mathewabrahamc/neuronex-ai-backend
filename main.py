import os
import re
from typing import Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app, origins="*")

# Validate and initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=api_key)

print(f"API Key: {api_key}")
print(f"Client: {client}")

@app.route("/evaluate", methods=["POST"])
def evaluate() -> Any:
    data: Dict[str, Any] = request.get_json()
    questions: Dict[str, Any] = data.get("questions", {})
    result: Dict[str, Any] = {
        "questionscore": {},
        "questionfeedback": {},
        "score": 0,
        "usage": {}
    }
    total_score = 0

    for qid, qdata in questions.items():
        answer = qdata.get("answer", "")
        questionDetails = qdata.get("questionDetails", {})
        evalConfig = qdata.get("evaluationConfig", {})

        modelAnswer = questionDetails.get("modelAnswer", "")
        questionText = questionDetails.get("questionText", "")
        max_score = evalConfig.get("max_score", 10)
        criteria = evalConfig.get("criteria", "")
        instructions = evalConfig.get("instructions", "")
        model = qdata.get("model", "gpt-3.5-turbo")
        max_tokens = qdata.get("max_tokens", 512)

        prompt = f"""
You are an examiner. Evaluate the student's answer based on the following:

🔹 Model Answer:
{modelAnswer}

🔹 Student Answer:
{answer}

🔹 Evaluation Criteria:
{criteria}

🔹 Instructions:
{instructions}

Provide:
Score: (out of {max_score})
Feedback: (One line improvement)
"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens
            )

            # Debug log for usage data
            print(f"🔍 Usage debug: {response.usage}")

            reply = response.choices[0].message.content

            score_match = re.search(r"Score\s*[:\-]?\s*(\d+)", reply)
            feedback_match = re.search(r"Feedback\s*[:\-]?\s*(.*)", reply)

            score = int(score_match.group(1)) if score_match else 0
            feedback = feedback_match.group(1).strip() if feedback_match else "No feedback"

            result["questionscore"][qid] = score
            result["questionfeedback"][qid] = feedback
            total_score += score

            # Safe extraction of token usage
            try:
                usage_data = response.usage
                result["usage"][qid] = {
                    "prompt_tokens": getattr(usage_data, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage_data, "completion_tokens", 0),
                    "total_tokens": getattr(usage_data, "total_tokens", 0)
                }
            except Exception as e:
                print(f"❌ Error extracting usage: {e}")
                result["usage"][qid] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }

        except Exception as e:
            result["questionscore"][qid] = 0
            result["questionfeedback"][qid] = f"Evaluation error: {str(e)}"
            result["usage"][qid] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

    result["score"] = total_score
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Result: {result}")
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
