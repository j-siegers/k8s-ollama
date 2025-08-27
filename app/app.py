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

    # Add a system prompt to instruct the model to be direct.
    # This can help prevent it from showing its internal reasoning.
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant. Do not include any internal thoughts, "
        "reasoning, or <think> tags in your answers. Only provide the direct answer to the user's question."
    }
    final_messages = [system_prompt] + messages

    def generate():
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": final_messages, "stream": True},
                stream=True,
            )
            response.raise_for_status()

            in_think_block = False
            for line in response.iter_lines():
                if not line:
                    continue

                decoded_line = line.decode('utf-8')
                try:
                    json_line = json.loads(decoded_line)
                    chunk = json_line.get('message', {}).get('content', '')

                    output_content = ""
                    while chunk:
                        if in_think_block:
                            end_index = chunk.find('</think>')
                            if end_index != -1:
                                in_think_block = False
                                chunk = chunk[end_index + len('</think>'):]
                            else:
                                # The rest of the chunk is inside the think block, discard it
                                chunk = ""
                        else:  # not in_think_block
                            start_index = chunk.find('<think>')
                            if start_index != -1:
                                # Part before the tag is valid
                                output_content += chunk[:start_index]
                                in_think_block = True
                                chunk = chunk[start_index + len('<think>'):]
                            else:
                                # No start tag, the whole chunk is valid
                                output_content += chunk
                                chunk = ""
                    if output_content:
                        json_line['message']['content'] = output_content
                        yield f"data: {json.dumps(json_line)}\n\n"
                except json.JSONDecodeError:
                    app.logger.warning(f"Could not decode line: {decoded_line}")
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Error connecting to Ollama: {e}")
            yield f"data: {json.dumps({'error': 'Could not connect to Ollama service.'})}\\n\n"

    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
