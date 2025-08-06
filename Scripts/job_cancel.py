from openai import OpenAI
import os

# Adding openAi job number below
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
job_id = ""  

# Cancelling job
result = client.fine_tuning.jobs.cancel(job_id)
print(f"Job {job_id} cancellation status: {result.status}")