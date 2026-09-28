from litellm import completion, completion_cost
import time
from dotenv import load_dotenv
load_dotenv()

def classify_task(user_query :str) -> str:
    """Cheap classifier — uses the fastest model to decide routing."""
    cls= completion(
        model = "groq/qwen/qwen3.8-27b",
        messages=[{
            "role":"user",
            "content":(
                f"Classify the following query into EXACTLY one word: "
                f"'code', 'summary', or 'general'. Query: {user_query}\n\nAnswer:"
            )
        }],
        max_tokens=5
        
    )
    return cls.choices[0].message.content.strip().lower()

def call_with_fallbacks(model_chain, messages):
    """Try each model in order; return the first one that succeeds."""
    last_error = None
    for model in model_chain:
        try:
            return completion(model=model, messages=messages)
        except Exception as e:
            print(f"{model} failed ({type(e).__name__}), trying next.....")
            last_error=e
            continue
    raise last_error

def smart_chat(user_query:str):
    """Routes to the right model based on task type, with fallbacks."""
    task = classify_task(user_query)
    
    routing={
        "code": ["groq/qwen/qwen3.8-27b", "groq/meta-llama/llama-4-scout-17b-16e-instruct"],
        "summary":["gemini/gemini-3.1-flash-lite", "groq/qwen/qwen3.8-27b"],
        "general":["gemini/gemini-3.1-flash-lite", "groq/qwen/qwen3.8-27b"]
    }
    
    model_chain =routing.get(task, routing["general"])
    
    start = time.time()
    response = call_with_fallbacks(
        model_chain= model_chain,
        messages= [{"role":"user","content":user_query}]
    )
    latency= time.time() -start
    try:
        cost = completion_cost(completion_response=response)
        cost_str = f"${cost:.6f}"
    except Exception:
        cost_str ="n/a"
    return {
        "detected_task":task,
        "model_used":response.model,
        "answer":response.choices[0].message.content,
        "latency_sec": round(latency,2),
        "cost_used":cost_str
    }    
    
def main():
    # Try it on three very different queries
    queries = [
        "Write a Python function to compute Fibonacci numbers.",
        "Summarize the importance of attention mechanism in 2 sentences.",
        "Tell me a fun fact about elephants."
    ]
    for q in queries:
        print("="*70)
        print("Q : ",q)
        result = smart_chat(q)
        print(f"Task:  {result['detected_task']}")
        print(f"Model:  {result['model_used']}")
        print(f"Latency:  {result['latency_sec']}")
        print(f"Cost:  {result['cost_used']}")
        print(f"Answer:  {result['answer']}")

if __name__ == "__main__":
    main()