from pymongo import MongoClient
import json

with open(r'C:\Users\amiri\Downloads\synthetic_sms_data.json', encoding='utf-8') as f:
    docs = [json.loads(line) for line in f if line.strip()]  # ← JSONL

client = MongoClient('mongodb://localhost:27017')
col = client['test']['contel']

for i in range(0, len(docs), 1000):
    col.insert_many(docs[i:i+1000])
    print(f"Inseré {min(i+1000, len(docs))}/{len(docs)}")

print(f"Total : {col.count_documents({})}")
client.close()