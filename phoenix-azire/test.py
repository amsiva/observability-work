from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(OPENROUTER_API_KEY)

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

embeddings = OpenAIEmbeddings(model="openai/text-embedding-3-small",
                              openai_api_key=OPENROUTER_API_KEY,
                              openai_api_base="https://openrouter.ai/api/v1")
print("Embedding client created successfully")

test = embeddings.embed_query("Hello world")
print("Embedding dimensions:", len(test))