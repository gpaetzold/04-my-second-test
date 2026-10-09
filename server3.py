from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import mlflow
from openai import AsyncOpenAI

# 1. MLflow & OpenAI-Autologging einrichten
mlflow.set_experiment("A2A-Local-LLM-Monitor")
mlflow.openai.autolog()  # Zeichnet alle LLM-Ein- und Ausgaben automatisch auf

# 2. Async-Client für das lokale Ollama initialisieren
llm_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama-local"  # Ein API-Key wird verlangt, der Wert ist bei Ollama aber egal
)

app = FastAPI(title="A2A Local LLM Server v3")

# --- Pydantic-Modelle für A2A ---
class MessageData(BaseModel):
    role: str
    text: str

class TaskRequest(BaseModel):
    task_id: str
    message: MessageData

class TaskResponsePayload(BaseModel):
    role: str
    text: str

class TaskResponse(BaseModel):
    task_id: str
    status: str
    original_message: str
    response: TaskResponsePayload


# --- Endpunkte ---

@app.get("/well-known/agent.json")
async def get_agent_card():
    return {
        "name": "Llama3.2 A2A Agent (v3)",
        "description": "Ein lokaler KI-Agent powered by Ollama, FastAPI und MLflow.",
        "url": "http://localhost:8000",
        "version": "3.0",
        "capabilities": {"streaming": False, "push_notifications": False}
    }

@app.post("/task/send", response_model=TaskResponse)
async def handle_task(payload: TaskRequest):
    # Ein übergeordneter Span klammert das gesamte A2A-Event im MLflow-UI
    with mlflow.start_span(name="A2A_Agent_Execution") as span:
        span.set_attribute("task_id", payload.task_id)
        
        system_instruction = (
            "Du bist ein hilfreicher KI-Agent in einem A2A (Agent-to-Agent) Netzwerk. "
            "Antworte präzise, höflich und halte dich kurz."
        )
        
        # Das lokale Modell asynchron aufrufen
        llm_response = await llm_client.chat.completions.create(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": payload.message.text}
            ],
            temperature=0.7
        )
        
        agent_text = llm_response.choices[0].message.content
        
        return TaskResponse(
            task_id=payload.task_id,
            status="completed",
            original_message=payload.message.text,
            response=TaskResponsePayload(
                role="agent",
                text=agent_text
            )
        )

if __name__ == '__main__':
    # WICHTIG: "server3:app" verweist nun korrekt auf diese Datei (server3.py)
    uvicorn.run("server3:app", host="0.0.0.0", port=8000, reload=True)
