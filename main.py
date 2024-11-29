# Importing the required libraries
import os
import re
import pickle
import warnings
import update_es
from tqdm import tqdm
from datetime import datetime
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from elasticsearch_dsl import Search, Q
print(f"Env variables loaded : {load_dotenv()}")
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import TokenTextSplitter

from llama_handler import LlamaChatHandler

hosts = "mock_host"
uname = "mock_user"
pswd = "mock_password"

es = Elasticsearch(hosts, request_timeout=60000, verify_certs=False, ssl_show_warn=False, basic_auth=(uname, pswd))
print(f"Connected to ES : {es.ping()}")

#---------------------------------------------------------------------------------------------------------------------------------------------------------#

# ElasticSearch Query to get desired mails
query_={
        "bool":{
            "must":[{"match_phrase":{'fileType':"EML"}},
                    {"match_phrase":{'tags':"High Priority"}},
                    {"match_phrase":{"boost.item_id_presence":True}}],
            
            "must_not":{"exists": {"field": "files_attached.MD5"}}
    }
}

res = Search(using=es, index=['ABC-2024*'])
res.query = Q(query_)
print(f"Total {res.count()} emails")

hit_list=list()
for hit in tqdm(res.scan()):
    hit_list.append(hit)
    if len(hit_list)==50:
        break

# Saving these mails in pickle for reusability
with open("high_priority_mails_2024.pkl", "wb")as file:
    pickle.dump(hit_list, file)

with open("high_priority_mails_2024.pkl", "rb")as file:
    hit_list = pickle.load(file)

#---------------------------------------------------------------------------------------------------------------------------------------------------------#

# Chunk size: The maximum number of characters or tokens allowed for each chunk. 

# Chunk overlap: Number of overlapping characters or tokens between chunks; overlapping chunks can help preserve cross-chunk context; the degree of overlap is typically specified as a percentage of the chunk size

model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()

def createChunks(page_text: str):
    try:
        text_splitter = TokenTextSplitter(encoding_name= "cl100k_base", chunk_size=1024, chunk_overlap=102)
        chunks = text_splitter.split_text(page_text)
        return chunks
    except Exception as e:
        return e

def stella_create_embedding(docs):
    global model
    doc_embeddings = model.encode(docs)
    return doc_embeddings
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------#

chunked_email_contents = []

for hit in tqdm(hit_list):
    content = hit['content']
    content = re.sub("\n+", "\n", content.replace("\r", " "))
    chunked_email_contents.append(createChunks(content))

content_embeddings = list()

for email_content in tqdm(chunked_email_contents):
    email_embeddings = stella_create_embedding(email_content)
    content_embeddings.append(email_embeddings)

#---------------------------------------------------------------------------------------------------------------------------------------------------------#

print("\n")
queries = [input("All these are High Priority mails, Type any query related to it...\n\n")]
query_embeddings = model.encode(queries, prompt_name = "s2p_query")

similarities_list = []

for index, embedding in tqdm(enumerate(content_embeddings)):
    similarities = model.similarity(query_embeddings, embedding)
    similarities_list.append([similarities.max(), hit_list[index]['metaData']['subject'], hit_list[index]['MD5'], index])

sorted_list = sorted(similarities_list, key=lambda x: x[0].item(), reverse=True)

# Printing the top 5 entries
top_5 = sorted_list[:5]
for entry in top_5:
    print(entry[1])

print()
user_input = int(input("Out of these 5 mails which one you want to analysis:\n"))

print(f"You selected mail of subject : {top_5[user_input-1][1]}")
print()

content = hit_list[top_5[user_input-1][-1]]['content'] #Fetching the content of mail selected by user
content = re.sub("\n+", "\n", content)
matches = re.findall(r'\b(abc\s*\d{11})\b', content) #Find all occurrences of the claim number pattern 


handler = LlamaChatHandler(model="llama3.1:8b-instruct-q5_1", temperature=0.3)  # instance of LlamaChatHandler class

# Analyze the email content using the specified claim numbers (converted to string).
# The method sends the email content and claim numbers to the LLM and returns a JSON response.
response = handler.analyze_email(content, str(matches))  
print(response['message']["content"])

mail_md5 = top_5[user_input-1][-2]
tag = response["status"]

#---------------------------------------------------------------------------------------------------------------------------------------------------------#

query = {
    "bool":{
        "must":[
            {"match_phrase_prefix": {"fileType" : "Email"}},
            {"match_phrase_prefix": {"MD5" : mail_md5}}]
    }
}

res = Search(using=es, index = ['ABC-2024*'])
res.query = Q(query)
res = res.source(['tags'])

file__ =  open('jsons/high_priority_mails.json', 'a')

for mail in res.scan():
    tags = mail["tags"] + [tag]

    try:
        # print(mail.meta.index)
        file__.write('{ "update": { "_index": "'+ mail.meta["index"] + '", "mail_id": "'+ mail.meta["id"] +'"} }\n')
        file__.write('{"doc": { "tags": ' + str(tags).replace("'", '"') + '} }\n')
    except Exception as someError:
        print("Exception ", mail.meta.index, mail.meta.id)
        print(someError)
file__.close()

update_es('jsons/high_priority_mails.json') # Function which is used to update ElasticSearch prod email data based on LLAMA Response