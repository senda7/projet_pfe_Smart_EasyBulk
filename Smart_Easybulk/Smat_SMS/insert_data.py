from pymongo import MongoClient
import json

with open(r'C:\Users\amiri\Downloads\contacts_synthetic.json', encoding='utf-8') as f:
    docs = json.load(f)

client = MongoClient('mongodb://localhost:27017')
col = client['test']['contel']

for i in range(0, len(docs), 1000):
    col.insert_many(docs[i:i+1000])
    print(f"Inseré {min(i+1000, len(docs))}/{len(docs)}")

print(f"Total : {col.count_documents({})}")
client.close()