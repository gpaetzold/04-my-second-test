from flask import Flask, jsonify, request
from datetime import datetime

app = Flask(__name__)

# 1. Agent Card - Definiert die Identität und Fähigkeiten des Agenten
AGENT_CARD = {
    "name": "Time Messenger Agent",
    "description": "Ein Agent, der auf Anfrage die aktuelle Uhrzeit zurückgibt.",
    "url": "http://localhost:5000",
    "version": "1.0",
    "capabilities": {
        "streaming": False,
        "push_notifications": False
    }
}

# http://localhost:5000/.well-known/agent.json
@app.route('/well-known/agent.json', methods=['GET'])
def get_agent_card():
    """Endpunkt zur Entdeckung des Agenten (Discovery Flow)."""
    return jsonify(AGENT_CARD)

# curl -X POST http://localhost:5000/task/send \
#   -H "Content-Type: application/json" \
#   -d '{"task_id": "123", "message": {"text": "What time is it?"}}'
@app.route('/task/send', methods=['POST'])
def handle_task():
    """Endpunkt zum Empfangen und Verarbeiten von A2A-Standardaufgaben."""
    data = request.get_json()
    
    # Extraktion der standardisierten A2A-Felder
    task_id = data.get("task_id")
    user_message = data.get("message", {}).get("text", "")
    
    # Logik des Agenten ausführen
    current_time = datetime.now().strftime("%H:%M:%S")
    response_text = f"Die aktuelle Uhrzeit ist {current_time}."
    
    # Standardisierte A2A-Antwortstruktur
    a2a_response = {
        "task_id": task_id,
        "status": "completed",
        "original_message": user_message,
        "response": {
            "role": "agent",
            "text": response_text
        }
    }
    
    return jsonify(a2a_response), 200

if __name__ == '__main__':
    # Startet den Server auf Port 5000
    app.run(host='0.0.0.0', port=5000)
