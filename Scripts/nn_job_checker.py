
import time
from openai import OpenAI
import os

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def poll_fine_tuning_job(job_id, interval_sec=60, max_wait_sec=3600):
    """
    Poll the status of a fine-tuning job every 60 seconds,
    up to 1 hour.
    Prints status updates and final result.
    """

    elapsed = 0
    while elapsed < max_wait_sec:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        print(f"Status: {status}")

        if status in ['succeeded', 'failed', 'cancelled']:
            print(f"Fine-tuning job finished with status: {status}")
            if status == 'succeeded':
                print(f"Fine-tuned model id: {job.fine_tuned_model}")
            else:
                if job.error:
                    print(f"Error info: {job.error.message}")
            return job

        time.sleep(interval_sec)
        elapsed += interval_sec

    print(f"Timed out waiting for job {job_id} to finish after {max_wait_sec} seconds.")
    return None

#fine_tune_job_id = "ftjob-vOkgTDViNP9madkHDyeZAkRm"
#fine_tune_job_id = "ftjob-FTcegVvIxoGzbhzC42iiPkpM"
#fine_tune_job_id = "ftjob-MWT3cwJI7BsJAjONwwpfFT0n"
#Last attempt fine_tune_job_id = "ftjob-xaYwZhLKnXJV3oJif5jARojO"
fine_tune_job_id = "ftjob-VVbCr2IIGrpp9L1ERdClBuT1"

final_job = poll_fine_tuning_job(fine_tune_job_id, interval_sec=60, max_wait_sec=3600)