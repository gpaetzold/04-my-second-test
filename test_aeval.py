import asyncio
import httpx
import json
import uuid
import sys

BASE_URL = "http://localhost:8000"
CONFIG_FILE = "eval.config.json"

async def run_aeval_pipeline():
    print("=== Starte AEVAL (Agentic Evaluation) Pipeline ===")
    
    # 1. Testvertrag laden
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Fehler: Konfigurationsdatei {CONFIG_FILE} nicht gefunden.")
        sys.exit(1)
        
    test_cases = config.get("test_cases", [])
    print(f"{len(test_cases)} Testfälle geladen.\n")
    
    passed_tests = 0
    
    # 2. Unabhängiger HTTP-Client für den Test-Harness
    async with httpx.AsyncClient(timeout=None) as client:
        for tc in test_cases:
            print(f"Running [{tc['id']}]...")
            print(f"Input: '{tc['input_text']}'")
            
            # Payload gemäß A2A-Spezifikation für server3.py vorbereiten
            payload = {
                "task_id": f"eval-{uuid.uuid4()}",
                "message": {
                    "role": "user",
                    "text": tc['input_text']
                }
            }
            
            # Ausführung über den Server (Executor-Rolle)
            try:
                response = await client.post(f"{BASE_URL}/task/send", json=payload)
                if response.status_code != 200:
                    print(f" -> FAILED: Server antwortete mit Status {response.status_code}")
                    continue
                    
                result_data = response.json()
                agent_output = result_data["response"]["text"]
                print(f"Agent Output: \"{agent_output.strip()}\"")
                
                # 3. Unabhängiges Grading (Grader-Rolle) - Deterministischer Keyword-Check
                missing_keywords = [
                    kw for kw in tc["required_keywords"] 
                    if kw.lower() not in agent_output.lower()
                ]
                
                if not missing_keywords:
                    print(" -> PASSED (Alle Kriterien erfüllt)\n")
                    passed_tests += 1
                else:
                    print(f" -> FAILED: Fehlende Schlüsselwörter: {missing_keywords}\n")
                    
            except Exception as e:
                print(f" -> FAILED aufgrund eines Ausnahmefehlers: {e}\n")
                
    # 4. Zusammenfassung für CI/CD-Gating
    print("=== AEVAL Zusammenfassung ===")
    print(f"Ergebnis: {passed_tests}/{len(test_cases)} Tests erfolgreich.")
    
    if passed_tests == len(test_cases):
        print("Status: SUCCESS 🎉")
        sys.exit(0)
    else:
        print("Status: FAILURE ❌ (Regressionsfehler erkannt!)")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(run_aeval_pipeline())
