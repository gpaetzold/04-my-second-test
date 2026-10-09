from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import mlflow
from openai import AsyncOpenAI
import json

# 1. MLflow einrichten
mlflow.set_experiment("A2A-Streaming-LLM-Monitor")
mlflow.openai.autolog()

# 2. Async-Ollama Client initialisieren
llm_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama-local"
)

app = FastAPI(title="A2A Streaming LLM Server v4")

# --- Pydantic-Modelle für A2A ---
class MessageData(BaseModel):
    role: str
    text: str

class TaskRequest(BaseModel):
    task_id: str
    message: MessageData

# --- Der asynchrone SSE-Generator ---
async def generate_llm_stream(task_id: str, prompt: str):
    """Ruft das LLM im Streaming-Modus auf und erzeugt ein SSE-konformes Protokoll."""
    system_instruction = (
        "Du bist ein hilfreicher KI-Agent in einem A2A Netzwerk. "
        "Antworte präzise, höflich und halte dich kurz."
    )
    
    # Aufruf mit stream=True
    llm_stream = await llm_client.chat.completions.create(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        stream=True  # WICHTIG: Aktiviert das stückchenweise Senden
    )
    
    # Wir iterieren asynchron über die eintreffenden Text-Fragmente (Chunks)
    async for chunk in llm_stream:
        # Prüfen, ob Text im Chunk enthalten ist
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            
            # SSE verlangt ein spezifisches Format: "data: <Inhalt>\n\n"
            # Um das A2A-Format zu wahren, verpacken wir das Token in ein JSON-Objekt
            chunk_payload = {
                "task_id": task_id,
                "status": "in_progress",
                "delta": token
            }
            yield f"data: {json.dumps(chunk_payload)}\n\n"
            
    # Finales Signal, dass der Stream beendet ist
    final_payload = {"task_id": task_id, "status": "completed", "delta": ""}
    yield f"data: {json.dumps(final_payload)}\n\n"


# --- Endpunkte ---

@app.get("/well-known/agent.json")
async def get_agent_card():
    return {
        "name": "Llama3.2 Streaming Agent (v4)",
        "description": "Ein lokaler KI-Agent mit Token-Streaming via SSE.",
        "url": "http://localhost:8000",
        "version": "4.0",
        "capabilities": {
            "streaming": True,  # HIER AKTIVIERT
            "push_notifications": False
        }
    }

@app.post("/task/send")
async def handle_task(payload: TaskRequest):
    """Gibt eine StreamingResponse anstelle eines statischen JSONs zurück."""
    # Wir übergeben den Generator an die FastAPI StreamingResponse
    return StreamingResponse(
        generate_llm_stream(payload.task_id, payload.message.text),
        media_type="text/event-stream"  # Der offizielle MIME-Type für SSE
    )

if __name__ == '__main__':
    # Pfad exakt angepasst an server4.py
    uvicorn.run("server4:app", host="0.0.0.0", port=8000, reload=True)
