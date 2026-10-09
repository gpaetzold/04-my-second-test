import os
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import mlflow
from openai import AsyncOpenAI
from dotenv import load_dotenv # NEU: dotenv laden

# .env Datei einlesen
load_dotenv()
LLM_MODEL = os.getenv("A2A_LLM_MODEL", "llama3.2:3b")

mlflow.set_experiment("A2A-Streaming-LLM-Monitor")
mlflow.openai.autolog()

llm_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama-local"
)

app = FastAPI(title="A2A Streaming LLM Server v4")

class MessageData(BaseModel):
    role: str
    text: str

class TaskRequest(BaseModel):
    task_id: str
    message: MessageData

async def generate_llm_stream(task_id: str, prompt: str):
    system_instruction = (
        "Du bist ein hilfreicher KI-Agent in einem A2A Netzwerk. "
        "Antworte präzise, höflich und halte dich kurz."
    )
    
    # HIER GEÄNDERT: Variable LLM_MODEL statt statischem String
    llm_stream = await llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        stream=True
    )
    
    async for chunk in llm_stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            chunk_payload = {
                "task_id": task_id,
                "status": "in_progress",
                "delta": token
            }
            yield f"data: {json.dumps(chunk_payload)}\n\n"
            
    final_payload = {"task_id": task_id, "status": "completed", "delta": ""}
    yield f"data: {json.dumps(final_payload)}\n\n"

@app.get("/well-known/agent.json")
async def get_agent_card():
    return {
        "name": f"Streaming Agent ({LLM_MODEL})",
        "description": "Ein lokaler KI-Agent mit konfigurierbarem Modell via .env.",
        "url": "http://localhost:8000",
        "version": "4.1",
        "capabilities": {"streaming": True, "push_notifications": False}
    }

@app.post("/task/send")
async def handle_task(payload: TaskRequest):
    return StreamingResponse(
        generate_llm_stream(payload.task_id, payload.message.text),
        media_type="text/event-stream"
    )

if __name__ == '__main__':
    uvicorn.run("server4:app", host="0.0.0.0", port=8000, reload=True)
