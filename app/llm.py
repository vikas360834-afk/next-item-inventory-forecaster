"""Optional OpenAI-compatible chat helper. Not required for forecasting."""
import os
import httpx

def chat(messages,base_url=None,api_key=None,model=None,**options):
    base_url=base_url or os.getenv("LLM_BASE_URL","http://localhost:11434/v1")
    api_key=api_key if api_key is not None else os.getenv("LLM_API_KEY","")
    model=model or os.getenv("LLM_MODEL","")
    if not model: raise RuntimeError("Set LLM_MODEL to use the optional chat helper.")
    headers={"Authorization":f"Bearer {api_key}"} if api_key else {}
    response=httpx.post(base_url.rstrip("/")+"/chat/completions",headers=headers,timeout=120,json={"model":model,"messages":messages,**options})
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
