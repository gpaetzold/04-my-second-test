import requests
import uuid

BASE_URL = "http://localhost:5000"

def run_client():
    # Phase 1: Discovery (Agent Card abrufen)
    print("Suche Agent Card...")
    card_response = requests.get(f"{BASE_URL}/well-known/agent.json")
    if card_response.status_code == 200:
        card = card_response.json()
        print(f"Agent gefunden: '{card['name']}' - {card['description']}\n")
    
    # Phase 2: Task senden (A2A-Kommunikation)
    task_payload = {
        "task_id": str(uuid.uuid4()),
        "message": {
            "role": "user",
            "text": "Wie spät ist es?"
        }
    }
    
    print("Sende Anfrage an den A2A-Server...")
    task_response = requests.post(f"{BASE_URL}/task/send", json=task_payload)
    
    if task_response.status_code == 200:
        result = task_response.json()
        print("Antwort vom Server-Agenten:")
        print(result["response"]["text"])

if __name__ == '__main__':
    run_client()
