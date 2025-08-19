
import os
import json
from flask import Flask, render_template, request, Response
import requests

app = Flask(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama-service:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:0.6b")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages", [])

    def generate():
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
                stream=True,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        json_line = json.loads(decoded_line)
                        if 'content' in json_line.get('message', {}):
                            yield f"data: {json.dumps(json_line)}\n\n"
                    except json.JSONDecodeError:
                        print(f"Could not decode line: {decoded_line}")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            yield f"data: {json.dumps({'error': 'Could not connect to Ollama service.'})}\\n\n"

    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
