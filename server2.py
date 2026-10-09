from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import mlflow

# MLflow initialisieren (Erstellt ein Experiment namens "A2A-Agent")
mlflow.set_experiment("A2A-Agent-Monitor")

# Aktiviert das automatische Tracing (besonders nützlich, wenn du später OpenAI/LangChain nutzt)
mlflow.autolog()

app = FastAPI(title="A2A Time Server v2 mit MLflow")

# --- Pydantic-Modelle ---
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

# --- Interne Logik-Funktion mit MLflow überwachen ---
@mlflow.trace(name="get_current_time_logic")
def get_time_logic(user_input: str) -> str:
    """Diese Funktion wird als eigener 'Span' im MLflow UI aufgezeichnet."""
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Optional: Du kannst manuelle Metriken oder Parameter im Trace loggen
    # mlflow.log_param("requested_by_user", user_input)
    
    return f"Die aktuelle Uhrzeit ist {current_time}."


# --- Endpunkte ---

@app.get("/well-known/agent.json")
async def get_agent_card():
    return {
        "name": "Time Messenger Agent (MLflow Enhanced)",
        "description": "Ein asynchroner Agent mit MLflow Monitoring.",
        "url": "http://localhost:8000",
        "version": "2.1",
        "capabilities": {"streaming": False, "push_notifications": False}
    }

@app.post("/task/send", response_model=TaskResponse)
async def handle_task(payload: TaskRequest):
    # Wir starten einen MLflow-Trace für diesen API-Aufruf
    with mlflow.start_span(name="A2A_Task_Endpoint") as span:
        # Metadaten an den Trace hängen
        span.set_attribute("task_id", payload.task_id)
        span.set_attribute("user_message", payload.message.text)
        
        # Aufruf der überwachten Logik
        response_text = get_time_logic(payload.message.text)
        
        span.set_attribute("agent_response", response_text)
        
        return TaskResponse(
            task_id=payload.task_id,
            status="completed",
            original_message=payload.message.text,
            response=TaskResponsePayload(
                role="agent",
                text=response_text
            )
        )

if __name__ == '__main__':
    uvicorn.run("server2:app", host="0.0.0.0", port=8000, reload=True)
