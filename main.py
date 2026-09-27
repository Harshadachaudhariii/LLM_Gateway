import warnings
import logging
from litellm import completion
import litellm
import os
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

litellm.suppress_debug_info = True




def main():
    print("Hello from llm-gateway!")
    print("Groq key loaded:   ","successfully loaded groq api key" if os.getenv("GROQ_API_KEY") else "Error in api key of groq")
    print("Gemini key loaded:   ","successfully loaded gemini api key" if os.getenv("GEMINI_API_KEY") else "Error in api key of gemini")
    print("Anthropic key loaded: ", "successfully loaded Anthropic api key" if os.getenv("ANTHROPIC_API_KEY") else "Error in api key of Anthropic")

if __name__ == "__main__":
    main()
