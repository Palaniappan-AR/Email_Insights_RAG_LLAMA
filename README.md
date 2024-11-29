# Email_Insights_RAG_LLAMA

# Overview
This project involves querying emails from Elasticsearch (ES), processing them to generate embeddings using the **Stella SentenceTransformer**, and conducting similarity searches based on user queries. The system utilizes **llama3.1:8b-instruct** for analyzing the selected email and provides JSON-formatted outputs. The final result updates email tags in ES based on the Llama response.

# Process Flow
**1.	Query Emails from Elasticsearch (ES)** <br />
  • Connects to Elasticsearch using specified host, username, and password.<br />
  • Queries for emails of type EML with the High Priority tag and missing MD5 in files_attached.<br />
  
**2.	Preprocess and Chunk Emails** <br />
  •	Retrieves emails from ES.<br />
  •	Preprocesses email content by removing extra newlines and replacing carriage returns.<br />
  •	Splits the email content into chunks using the TokenTextSplitter from LangChain to ensure context preservation across chunks.<br />
  
**3.	Generate Embeddings** <br />
  •	Utilizes Stella SentenceTransformer to generate embeddings for each chunk of email content.<br />
  •	Stores the embeddings in a list for further processing.<br />
  
**4.	User Query and Similarity Search** <br />
  •	Accepts a user-defined query related to high-priority emails.<br />
  •	Generates embeddings for the query and compares it with email embeddings using a similarity search.<br />
  •	Sorts and displays the top 5 most relevant emails based on similarity scores.<br />
  
**5.	Email Analysis with Llama** <br />
  •	Prompts the user to select one of the top 5 emails for further analysis.<br />
  •	Extracts claim numbers from the email content using a regex pattern.<br />
  •	Sends the email content and claim numbers to Llama using the LlamaChatHandler class.<br />
  •	The Llama model uses prompt engineering and few-shot learning to generate a JSON-formatted response.<br />
  
**6.	Save and Update Tags in Elasticsearch** <br />
  •	Saves the Llama response in a JSON file.<br />
  •	Updates the email’s tags in Elasticsearch based on the Llama response using a batch update process.<br />

# File Descriptions
  •	high_priority_mails_2024.pkl: A pickle file that stores the queried email list for reusability.<br />
  •	jsons/high_priority_mails.json: A JSON file that stores the Llama-generated email responses for ES updates.

# Execute the Script to:
  •	Query Elasticsearch for relevant emails.<br />
  •	Preprocess, chunk, and generate embeddings.<br />
  •	User gives prompt for queries and fetch relevant results.<br />
  •	Send the selected email for Llama analysis and update Elasticsearch.
  
# Updating Elasticsearch:
  • The update_es function reads the JSON file and updates the email tags in ES.
