import asyncio
import httpx
import uuid

BASE_URL = "http://localhost:8000"

async def run_client():
    # timeout=None verhindert den ReadTimeout-Fehler bei langsameren LLM-Generierungen
    async with httpx.AsyncClient(timeout=None) as client:
        
        # Phase 1: Discovery (Agent Card abrufen)
        print("Suche Agent Card...")
        card_response = await client.get(f"{BASE_URL}/well-known/agent.json")
        if card_response.status_code == 200:
            card = card_response.json()
            print(f"Agent gefunden: '{card['name']}' - {card['description']}\n")
        
        # Phase 2: Task payload vorbereiten
        task_payload = {
            "task_id": str(uuid.uuid4()),
            "message": {
                "role": "user",
                "text": "Erkläre kurz, was ein A2A Server ist."
            }
        }
        
        print("Sende Anfrage an den A2A-Server...")
        task_response = await client.post(f"{BASE_URL}/task/send", json=task_payload)
        
        if task_response.status_code == 200:
            result = task_response.json()
            print("Antwort vom Server-Agenten:\n")
            print(result["response"]["text"])
        else:
            print(f"Fehler: {task_response.status_code}")
            print(task_response.text)

if __name__ == '__main__':
    asyncio.run(run_client())
