import asyncio
import httpx
import json
import uuid
import sys
import os  # NEU
import mlflow
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv  # NEU

# .env Datei einlesen
load_dotenv()
LLM_MODEL = os.getenv("A2A_LLM_MODEL", "llama3.2:3b")

BASE_URL = "http://localhost:8000"
CONFIG_FILE = "eval2.config.json"

mlflow.set_experiment("A2A-Streaming-AEVAL-Pipeline Linux")

judge_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama-judge"
)

async def evaluate_with_llm(input_text: str, agent_output: str, criteria: str) -> tuple[bool, str]:
    """Nutzt das konfigurierte lokale LLM mit Chain-of-Thought für die Bewertung."""
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
        # HIER GEÄNDERT: Variable LLM_MODEL genutzt
        response = await judge_client.chat.completions.create(
            model=LLM_MODEL,
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
    print(f"=== Starte AEVAL Streaming-Pipeline mit {CONFIG_FILE} ===")
    print(f"Genutztes Richter-Modell: {LLM_MODEL}\n")
    
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Fehler: Konfigurationsdatei {CONFIG_FILE} nicht gefunden.")
        sys.exit(1)
        
    test_cases = config.get("test_cases", [])
    print(f"{len(test_cases)} Testfälle geladen.\n")
    
    passed_tests = 0
    
    suite_name = f"AEVAL_Streaming_Suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=suite_name):
        # MLflow den genutzten Modellnamen als globalen Parameter übergeben
        mlflow.log_param("evaluated_model", LLM_MODEL)
        
        async with httpx.AsyncClient(timeout=None) as client:
            for tc in test_cases:
                print(f"Running [{tc['id']}]...")
                print(f"Input: '{tc['input_text']}'")
                
                payload = {
                    "task_id": f"eval-{uuid.uuid4()}",
                    "message": {"role": "user", "text": tc['input_text']}
                }
                
                full_agent_output = ""
                
                try:
                    async with client.stream("POST", f"{BASE_URL}/task/send", json=payload) as response:
                        if response.status_code != 200:
                            print(f" -> FAILED: Server-Status {response.status_code}")
                            continue
                        
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if line.startswith("data: "):
                                json_str = line[6:]
                                try:
                                    chunk_data = json.loads(json_str)
                                    full_agent_output += chunk_data.get("delta", "")
                                except json.JSONDecodeError:
                                    continue
                    
                    full_agent_output = full_agent_output.strip()
                    print(f"Agent Output (rekonstruiert): \"{full_agent_output}\"")
                    
                    if not full_agent_output:
                        print(" -> FAILED: Es wurden keine Streaming-Daten rekonstruiert.\n")
                        continue
                    
                    print("Rufe LLM-Richter auf...")
                    is_valid, judge_feedback = await evaluate_with_llm(
                        input_text=tc['input_text'],
                        agent_output=full_agent_output,
                        criteria=tc['evaluation_criteria']
                    )
                    
                    with mlflow.start_run(run_name=tc['id'], nested=True):
                        mlflow.log_param("test_id", tc['id'])
                        mlflow.log_param("input_text", tc['input_text'])
                        mlflow.log_param("criteria", tc['evaluation_criteria'])
                        
                        mlflow.log_text(full_agent_output, f"{tc['id']}_agent_output.txt")
                        mlflow.log_text(judge_feedback, f"{tc['id']}_judge_feedback.txt")
                        
                        if is_valid:
                            print(" -> TEST-STATUS: PASSED ✅\n")
                            passed_tests += 1
                            mlflow.log_metric("passed", 1)
                        else:
                            print(" -> TEST-STATUS: FAILED ❌\n")
                            mlflow.log_metric("passed", 0)
                        
                except Exception as e:
                    print(f" -> FAILED aufgrund eines Ausnahmefehlers: {e}\n")
                    
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
    asyncio.run(run_aeval_pipeline())
