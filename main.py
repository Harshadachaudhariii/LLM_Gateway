import warnings, logging, os ,re, time
from litellm import completion, completion_cost, Router
import litellm
from litellm.caching import Cache 
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

litellm.suppress_debug_info = True

def response_llm():

    prompt = "Explain RAG in one sentence."
    provider=[
            ("GEMINI", "gemini/gemini-3.1-flash-lite"),
            ("GROQ", "groq/qwen/qwen3.8-27b"),
            ("ANTHROPIC","claude-3-5-haiku-20241022")
        ]
    for label, models in provider:
        try:
            response_llm =completion(
                model=models, messages=[{"role":"user","content":prompt}]
            )
            print(f"{label:<15}: {response_llm.choices[0].message.content[:50]}")

        except Exception as e:
            print({type(e).__name__})

def fallbacks():
    response = completion(
        model="claude-3-5-haiku-20241022",
        messages=[{"role":"user", "content":"what is LLM gateway"}],
        fallbacks=[
            "gemini/gemini-3.1-flash-lite",
            "groq/qwen/qwen3.8-27b"
        ]
    )
    print("Response: ", response.choices[0].message.content[:100])
    print("\nWhich model actually response?", response.model)

def cost_tracking():
    response = completion(
        model = "groq/qwen/qwen3.8-27b",
        messages=[{"role":"user", "content":"what is LLM gateway"}]
    )
    cost = completion_cost(completion_response=response)
    
    print("Response:    ", response.choices[0].message.content)
    print("\nInput tokens: ", response.usage.prompt_tokens)
    print("Output tokens:", response.usage.completion_tokens)
    print(f"Cost:         ${cost:.8f}")

def llm_caching():
    litellm.callbacks =[]
    litellm.success_callback =[]
    litellm.failure_callback =[]
    litellm._async_success_callback=[]
    litellm._async_failure_callback =[]

    litellm.cache=None
    print("LiteLLM state reset - ready for clean caching demo")
    
    litellm.cache = Cache(type="local")
    prompt = "What does LLM stand for? Answer in one line"
    
    start = time.time()
    response1 = completion(
        model = "gemini/gemini-3.1-flash-lite",
        messages=[{"role":"user","content":prompt}],
        caching =True
    )
    time1= time.time() -start
    print(f"First API call: {time1:2f}s - {response1.choices[0].message.content}")
    
    start = time.time()
    response2 = completion(
        model="gemini/gemini-3.1-flash-lite",
        messages=[{"role":"user","content":prompt}],
        caching =True
    )
    time2 = time.time()-start
    print(f"Second API call: {time2:2f}s -{response2.choices[0].message.content}")
    print(f"\n Speedup: {time1/time2:.1f}x faster, and zero cost on the second call!")
    
def routing():
    model_list =[
        {
            "model_name":"fast-cheap",
            "litellm_params":{
                "model": "gemini/gemini-3.1-flash-lite",
                "api_key":os.getenv("GEMINI_API_KEY")
            }
        },
        {
            "model_name":"smart-coding",
            "litellm_params":{
                "model":"groq/qwen/qwen3.8-27b",
                "api_key":os.getenv("GROQ_API_KEY")
            }
        },
        {
            "model_name":"balanced",
            "litellm_params":{
                "model":"groq/meta-llama/llama-4-scout-17b-16e-instruct",
                "api_key":os.getenv("GROQ_API_KEY")
            }
        }
    ]
    
    router =Router(model_list=model_list)
    
    fast_response = router.completion(
        model ="fast-cheap",
        messages=[{"role":"user","content":"Summarize: AI is changing software."}]
    )
    
    code_response = router.completion(
        model = "smart-coding",
        messages=[{"role":"user","content":"Write a Python function to reverse a string."}]
    )
    
    print("Fast/cheap (Groq): ", fast_response.choices[0].message.content[:100])
    print("\nsmart/cosing (Gemini): ", code_response.choices[0].message.content[:300])
    
def load_balancing():
    model_list =[
            {
                "model_name":"gpt-pool",
                "litellm_params":{
                    "model": "gemini/gemini-3.1-flash-lite",
                    "api_key":os.getenv("GEMINI_API_KEY")
                },
                "model_info":{"id":"gemini-flash-lite"}
            },
            {
                "model_name":"gpt-pool",
                "litellm_params":{
                    "model":"groq/qwen/qwen3.8-27b",
                    "api_key":os.getenv("GROQ_API_KEY")
                },
                "model_info":{"id":"groq-qwen-27b"}
            }
        ]
    router = Router(
        model_list=model_list,
        routing_strategy="simple-shuffle"
    )
    
    print(f"{'Request':<10}{'Deployment Picked':<22}{'Latency':<12}{'Response':<40}")
    print("-" * 84)
    for i in range(6):
        r = router.completion(
            model ="gpt-pool",
            messages=[{"role":"user", "content":f"Say hello, request{i+1}"}]
        )
        deployment_id = r._hidden_params.get("model_id", "unknown")
        latency = r._response_ms
        answer= r.choices[0].message.content[:35]
        print(f"#{i+1:<9}{deployment_id:<22}{latency:>6.0f} ms {answer}")        
    
def gateway_with_langchain():
    llm= ChatLiteLLM(model="groq/qwen/qwen3.8-27b", temperature=0.3)
    
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI tutor named HarshadaAI. Be concise."),
        ("user", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question":"what is an LLM Gateway in 3 bullets?"})
    print(answer)
   
def gateway_with_langchain_with_multi_provider():
    primary = ChatLiteLLM(model="groq/qwen/qwen3.8-27b")
    
    fallbacks_1 = ChatLiteLLM(model="gemini/gemini-3.1-flash-lite", temperature=0.2)
    fallbacks_2 = ChatLiteLLM(model="gemini/gemini-3.1-flash-lite", temperature=0.2)
    
    robust_llm = primary.with_fallbacks([fallbacks_1, fallbacks_2])
    
    prompt= ChatPromptTemplate.from_messages([
        ("system","You are an expert AI engineer. Always reply in JSON: {{\"answer\": ...}}"),
        ("user", "{questions}")
    ])
    
    chain = prompt | robust_llm | StrOutputParser()
    
    result = chain.invoke({"questions": "What are the top 3 benefits of an LLM Gateway?"})
    print(result)
    
# ----------------------------------------gaurdrails--------------------------------------------
PII_PATTERNS = {
    "EMAIL":       r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE_IN":    r"(\+91[\-\s]?)?[6-9]\d{9}",                  # Indian mobile
    "PHONE_US":    r"(\+1[\-\s]?)?\(?\d{3}\)?[\-\s]?\d{3}[\-\s]?\d{4}",
    "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
    "AADHAAR":     r"\b\d{4}\s?\d{4}\s?\d{4}\b",                 # Indian Aadhaar
    "PAN":         r"\b[A-Z]{5}\d{4}[A-Z]\b",                    # Indian PAN
    "CREDIT_CARD": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    "IP_ADDRESS":  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }
def redact_pii(text: str):
    """Replace PII in text with placeholders. Returns (clean_text, detected_list)."""
    detected = []
    clean = text
    for label, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, clean)
        if matches:
            detected.append({"type": label, "count": len(matches)})
            clean = re.sub(pattern, f"<{label}_REDACTED>", clean)
    return clean, detected

def pii_input_guardrail(kwargs):
    """LiteLLM pre-call hook: scrub PII from user messages."""
    messages = kwargs.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            clean, detected = redact_pii(msg["content"])
            if detected:
                print(f"PII REDACTED: {detected}")
                msg["content"] = clean
# Register the guardrail
def gaurdrail():
    litellm.input_callback = [pii_input_guardrail]
    user_msg = (
        "Hi, I'm Miss. Harshada. My email is krish@krishnaik.in, "
        "my Indian mobile is +91-9876543210, my PAN is ABCDE1234F, "
        "and my Aadhaar is 1234 5678 9012. Help me write Python code."
    )

    response = completion(
        model="groq/qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": user_msg}],
        max_tokens=80
    )
    print("\nLLM Response:")
    print(response.choices[0].message.content)

# ---------------------------------------gaurdrail finished---------------------------------
    
def main():
    print("Hello from llm-gateway!")
    print("Groq key loaded:   ","successfully loaded groq api key" if os.getenv("GROQ_API_KEY") else "Error in api key of groq")
    print("Gemini key loaded:   ","successfully loaded gemini api key" if os.getenv("GEMINI_API_KEY") else "Error in api key of gemini")
    print("Anthropic key loaded: ", "successfully loaded Anthropic api key" if os.getenv("ANTHROPIC_API_KEY") else "Error in api key of Anthropic")
    
    # response_llm()
    # fallbacks()
    # cost_tracking()
    # llm_caching()
    # routing()
    # load_balancing()
    # gateway_with_langchain()
    # gateway_with_langchain_with_multi_provider()
    gaurdrail()

if __name__ == "__main__":
    main()
