import asyncio
import httpx
import json
import uuid
import sys
import mlflow  # NEU: MLflow importieren
from openai import AsyncOpenAI

BASE_URL = "http://localhost:8000"
CONFIG_FILE = "eval2.config.json"

# MLflow Experiment für die Test-Pipeline definieren
mlflow.set_experiment("A2A-AEVAL-Pipeline")

# Der unbestechliche Richter-Client für das lokale Ollama
judge_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama-judge"
)

async def evaluate_with_llm(input_text: str, agent_output: str, criteria: str) -> tuple[bool, str]:
    """Nutzt ein lokales LLM, bewertet die Qualität und gibt (Status, Begründung) zurück."""
    judge_prompt = f"""
    Du bist ein präziser Qualitätsprüfer für KI-Software.
    Analysiere die Antwort des Agenten basierend auf dem vorgegebenen Kriterium.

    URSPRÜNGLICHE ANFRAGE DES USERS: {input_text}
    TATSÄCHLICHE ANTWORT DES AGENTEN: {agent_output}
    ZUR PRÜFUNG STEHENDES KRITERIUM: {criteria}

    Schritt 1: Begründe kurz in einem Satz, ob die Antwort das Kriterium inhaltlich erfüllt.
    Schritt 2: Beende deine Antwort zwingend mit dem exakten Label: [ERGEBNIS: PASSED] oder [ERGEBNIS: FAILED]
    """
    
    try:
        response = await judge_client.chat.completions.create(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0
        )
        
        verdict = response.choices[0].message.content.strip()
        print(f" -> Richter-Begründung:\n    {verdict}")
        
        is_passed = "[ERGEBNIS: PASSED]" in verdict
        return is_passed, verdict
        
    except Exception as e:
        print(f" -> Richter-Fehler: {e}")
        return False, str(e)

async def run_aeval_pipeline():
    print(f"=== Starte AEVAL Pipeline mit {CONFIG_FILE} ===")
    
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Fehler: Konfigurationsdatei {CONFIG_FILE} nicht gefunden.")
        sys.exit(1)
        
    test_cases = config.get("test_cases", [])
    print(f"{len(test_cases)} Testfälle geladen.\n")
    
    passed_tests = 0
    
    # NEU: Wir starten einen übergeordneten MLflow-Run für diesen gesamten Testdurchlauf
    with mlflow.start_run(run_name=f"AEVAL_Suite_{datetime.now().strftime('%Y%m%d_%H%M%S') if 'datetime' in globals() else uuid.uuid4().hex[:6]}"):
        
        async with httpx.AsyncClient(timeout=None) as client:
            for tc in test_cases:
                print(f"Running [{tc['id']}]...")
                print(f"Input: '{tc['input_text']}'")
                
                payload = {
                    "task_id": f"eval-{uuid.uuid4()}",
                    "message": {"role": "user", "text": tc['input_text']}
                }
                
                try:
                    # 1. Ausführung über den Server (Executor)
                    response = await client.post(f"{BASE_URL}/task/send", json=payload)
                    if response.status_code != 200:
                        print(f" -> FAILED: Server-Status {response.status_code}")
                        continue
                        
                    result_data = response.json()
                    agent_output = result_data["response"]["text"]
                    print(f"Agent Output: \"{agent_output.strip()}\"")
                    
                    # 2. Semantische Bewertung durch den LLM-Judge
                    is_valid, judge_feedback = await evaluate_with_llm(
                        input_text=tc['input_text'],
                        agent_output=agent_output,
                        criteria=tc['evaluation_criteria']
                    )
                    
                    # NEU: Wir nutzen MLflow Nested Spans / Child Logs, um jeden Testfall einzeln zu protokollieren
                    with mlflow.start_run(run_name=tc['id'], nested=True):
                        # Parameter zur Nachvollziehbarkeit loggen
                        mlflow.log_param("test_id", tc['id'])
                        mlflow.log_param("input_text", tc['input_text'])
                        mlflow.log_param("criteria", tc['evaluation_criteria'])
                        
                        # Ausgaben als Text-Inhalte hinterlegen
                        mlflow.log_text(agent_output, f"{tc['id']}_agent_output.txt")
                        mlflow.log_text(judge_feedback, f"{tc['id']}_judge_feedback.txt")
                        
                        if is_valid:
                            print(" -> TEST-STATUS: PASSED ✅\n")
                            passed_tests += 1
                            mlflow.log_metric("passed", 1)  # 1 = Bestanden
                        else:
                            print(" -> TEST-STATUS: FAILED ❌\n")
                            mlflow.log_metric("passed", 0)  # 0 = Durchgefallen
                        
                except Exception as e:
                    print(f" -> FAILED aufgrund eines Ausnahmefehlers: {e}\n")
                    
        # Gesamtergebnis für die gesamte Test-Suite im übergeordneten Run loggen
        mlflow.log_metric("total_tests", len(test_cases))
        mlflow.log_metric("passed_tests", passed_tests)
        mlflow.log_metric("success_rate", passed_tests / len(test_cases) if test_cases else 0)
                    
    print("=== AEVAL Zusammenfassung ===")
    print(f"Ergebnis: {passed_tests}/{len(test_cases)} Tests erfolgreich.")
    
    if passed_tests == len(test_cases):
        print("Status: SUCCESS 🎉")
        sys.exit(0)
    else:
        print("Status: FAILURE ❌")
        sys.exit(1)

if __name__ == '__main__':
    # Hilfsimport für den Zeitstempel im Run-Namen
    from datetime import datetime
    asyncio.run(run_aeval_pipeline())
