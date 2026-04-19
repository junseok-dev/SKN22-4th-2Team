import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
index_name = "patent-guard-hybrid"

print(f"Connecting to Pinecone with index: {index_name}")

try:
    pc = Pinecone(api_key=api_key)
    indexes = pc.list_indexes()
    print(f"Available indexes: {[i.name for i in indexes]}")
    
    found = False
    for i in indexes:
        if i.name == index_name:
            print(f"Index '{index_name}' found!")
            print(f"Dimension: {i.dimension}")
            print(f"Metric: {i.metric}")
            print(f"Spec: {i.spec}")
            found = True
            break
            
    if not found:
        print(f"Index '{index_name}' NOT found. The backend will try to create it.")

except Exception as e:
    print(f"Error connecting to Pinecone: {e}")
