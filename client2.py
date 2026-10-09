import asyncio
import httpx
import uuid

BASE_URL = "http://localhost:8000"

async def run_client():
    # Wir nutzen httpx für asynchrone HTTP-Requests
    async with httpx.AsyncClient() as client:
        
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
                "text": "Wie spät ist es?"
            }
        }
        
        print("Sende Anfrage an den A2A-Server...")
        task_response = await client.post(f"{BASE_URL}/task/send", json=task_payload)
        
        if task_response.status_code == 200:
            result = task_response.json()
            print("Antwort vom Server-Agenten:")
            print(result["response"]["text"])
        else:
            print(f"Fehler: {task_response.status_code}")
            print(task_response.text)

if __name__ == '__main__':
    # Startet die asynchrone Event-Loop für den Client
    asyncio.run(run_client())
