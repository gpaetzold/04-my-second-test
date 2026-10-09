import asyncio
import httpx
import uuid
import json
import sys

BASE_URL = "http://localhost:8000"

async def run_client():
    # Unbegrenztes Timeout, da die Verbindung permanent streamt
    async with httpx.AsyncClient(timeout=None) as client:
        
        # Phase 1: Discovery (Agent Card abrufen)
        print("Suche Agent Card...")
        card_response = await client.get(f"{BASE_URL}/well-known/agent.json")
        if card_response.status_code == 200:
            card = card_response.json()
            print(f"Agent gefunden: '{card['name']}' (Streaming: {card['capabilities']['streaming']})\n")
        
        # Phase 2: Payload vorbereiten
        task_payload = {
            "task_id": str(uuid.uuid4()),
            "message": {
                "role": "user",
                "text": "Schreibe ein kurzes Gedicht über künstliche Intelligenz."
            }
        }
        
        print("Sende Anfrage an den Streaming A2A-Server...")
        print("Antwort vom Server-Agenten:\n---")
        
        # Mit client.stream('POST', ...) öffnen wir die Standleitung für SSE
        async with client.stream("POST", f"{BASE_URL}/task/send", json=task_payload) as response:
            if response.status_code != 200:
                print(f"Fehler: {response.status_code}")
                return

            # Wir lesen den Stream Zeile für Zeile aus
            async for line in response.aiter_lines():
                # SSE-Nachrichten beginnen standardmäßig mit "data: "
                if line.startswith("data: "):
                    # Extrahiere das reine JSON-Objekt hinter "data: "
                    json_str = line[6:]
                    try:
                        chunk_data = json.loads(json_str)
                        token = chunk_data.get("delta", "")
                        
                        # Token sofort ohne Zeilenumbruch im Terminal ausgeben
                        sys.stdout.write(token)
                        sys.stdout.flush()  # Erzwingt die sofortige Anzeige im Terminal
                        
                        if chunk_data.get("status") == "completed":
                            print("\n---")
                            print("Stream erfolgreich beendet.")
                    except json.JSONDecodeError:
                        continue

if __name__ == '__main__':
    asyncio.run(run_client())
