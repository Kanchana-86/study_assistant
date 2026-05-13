import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Scholarly, an expert student study assistant.
Your goal is to make complex topics crystal clear.

Guidelines:
- Explain concepts in plain, simple English
- Use short, punchy sentences
- Use bullet points for lists and steps
- Bold key terms using **term**
- Give real-world examples when helpful
- Be encouraging and friendly
- Keep responses focused and concise
"""

# ─── GROQ CHAT FUNCTION ────────────────────────────────────────────────────────
def ask_groq(question, history=None):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include previous messages for context (last 10 exchanges)
    if history:
        for msg in history[-10:]:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        result = response.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]

        print("API Error:", result)
        return "Error getting response from AI. Please try again."

    except Exception as e:
        print("Request failed:", e)
        return f"Request failed: {str(e)}"


# ─── PDF EXTRACTION ────────────────────────────────────────────────────────────
def extract_pdf_text(file):
    try:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

        return text.strip()

    except Exception as e:
        print("PDF Error:", e)
        return ""


# ─── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    history = data.get("history", [])  # Optional chat history for context

    if not question:
        return jsonify({"answer": "Please enter a question."})

    answer = ask_groq(question, history)
    return jsonify({"answer": answer})


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"answer": "No file uploaded."})

    file = request.files["file"]

    if not file.filename.endswith(".pdf"):
        return jsonify({"answer": "Please upload a valid PDF file."})

    text = extract_pdf_text(file)

    if not text.strip():
        return jsonify({"answer": "Could not read the PDF. It may be scanned or image-based."})

    prompt = (
        "Summarize this document for a student. "
        "Highlight the key points, main ideas, and any important terms. "
        "Make it easy to understand:\n\n"
        + text[:4000]
    )

    answer = ask_groq(prompt)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
