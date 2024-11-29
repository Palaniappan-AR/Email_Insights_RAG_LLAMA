import ollama

schema = { 
  "claim_number": {
    "type": "string",
    "description": "Claim number from email body"
  },
  "status": {
    "type": "string",
    "description": "The current status of the claim, which must be one of the following: 'approved', 'denied', 'partially approved', or 'pending'."
  },
  "reason": {
    "type": "string",
    "description": "Text phrase from the email that influenced the status determination"
  }
}

schema_text_1 = "Email Body and Metadat of mail 1"

schema_output_1 = """[
  {
    "claim_number": "238724",
    "status": "denied",
    "reason": "these items were not featured on an sale"
  },
  {
    "claim_number": "982451",
    "status": "denied",
    "reason": "we did not offer funding"
  }
]"""

schema_text_2 = "Email Body and Metadat of mail 2"

schema_output_2 = """[
  {
    "claim_number": "248325",
    "status": "approved",
    "reason": "this is approved now."
  },
  {
    "claim_number": "984215",
    "status": "approved",
    "reason": "this is approved now."
  }
]"""

schema_text_3 = "Email Body and Metadat of mail 3"

schema_output_3 = """[
  {
    "claim_number": "237842",
    "status": "denied",
    "reason": "this is denied as collected in q3 2024"
  }
]"""

class LlamaChatHandler:  
    def __init__(self, model="llama3.1:8b-instruct-q5_1", temperature=0.2):  
        self.model = model
        self.temperature = temperature

    def analyze_email(self, content: str, matches: str) -> str:
        content = "Email Content:\n\n" + content + "\n\nClaim Numbers:\n" + matches + "\n\nPlease analyze the email conversation and provide the status of the specified claim numbers."

        messages = [
            {"role": "system", "content": f"""You are an expert in email sentiment analysis. Your task is to analyze email conversations between a abc.com and any vendor to determine the status of given claim numbers. Focus on identifying key sentiments, updates, and relevant details regarding each claim's progress. 
                                            Output in JSON using the schema defined here: {schema}, If there are multiple IRA numbers, provide a list of JSON objects. Do not include any additional information; only return the specified JSON."""},
            {"role": "user", "content": schema_text_1},
            {"role": "assistant", "content": schema_output_1},
            {"role": "user", "content": schema_text_2},
            {"role": "assistant", "content": schema_output_2},
            {"role": "user", "content": schema_text_3},
            {"role": "assistant", "content": schema_output_3},
            {"role": "user", "content": content}
        ]
        
        try:
          response = ollama.chat(model = self.model, messages = messages, format = "json", stream = False, options={"temperature":0.2})
        except Exception as e:
            return f"Error occured : {str(e)}"
        return response['message']['content']