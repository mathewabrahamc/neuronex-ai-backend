import os
import openai
from flask import Flask, request, jsonify
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    questions = data.get("questions", {})
    result = {"questionscore": {}, "questionfeedback": {}, "score": 0}
    total_score = 0

    for qid, qdata in questions.items():
        answer = qdata.get("answer", "")
        questionDetails = qdata.get("questionDetails", {})
        evalConfig = qdata.get("evaluationConfig", {})
        modelAnswer = questionDetails.get("modelAnswer", "")
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
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            reply = response.choices[0].message.content
            score_match = re.search(r"Score\\s*[:\\-]?\\s*(\\d+)", reply)
            feedback_match = re.search(r"Feedback\\s*[:\\-]?\\s*(.*)", reply)

            score = int(score_match.group(1)) if score_match else 0
            feedback = feedback_match.group(1).strip() if feedback_match else "No feedback"

            result["questionscore"][qid] = score
            result["questionfeedback"][qid] = feedback
            total_score += score
        except Exception as e:
            result["questionscore"][qid] = 0
            result["questionfeedback"][qid] = f"Evaluation error: {str(e)}"

    result["score"] = total_score
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
