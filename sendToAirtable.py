import requests
import os
from dotenv import load_dotenv
load_dotenv()
import json

def add_new_record(imageURL,promptName,userName):
    AIRTABLE_TOKEN = "key5fg2TuOURbROQ3"
    AIRTABLE_BASE_ID = "appSmxUStD4yA533I"
    AIRTABLE_TABLE_NAME = "ImageData"
    AIRTABLE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"

    """Add record to the Airtable."""

    headers = {
      'Authorization': f'Bearer {AIRTABLE_TOKEN}',
      'Content-Type': 'application/json'
    }
    new_data = {
        "records": [
            {
                "fields": {
                    "ImageURL": f"{imageURL}",
                    "PromptName": f"{promptName}",
                    "UserName": f"{userName}"
                }
            }
        ]
    }

    response = requests.request("POST", AIRTABLE_URL, headers=headers, data=json.dumps(new_data))

    return response