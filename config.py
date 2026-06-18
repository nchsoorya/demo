import json
import os

# =====================================================
# SCHEMA DEFINITION
# =====================================================
SCHEMA = {
    "First_Name": "None",
    "Last_Name": "None",
    "Salutation": "None",
    "Gender": "None",
    "Date_Of_Birth": "None",
    "Birth_City": "None",
    "Birth_Province_or_State": "None",
    "Birth_Country": "None",
    "Tax_Code": "None",
    "Citizenship_Country": "None",
    "Marital_Status": "None",
    "ID_Info": {
        "Passport_Number": "None",
        "ID_Card_Number": "None",
        "Document_Issue_Date": "None",
        "Document_Expiry_Date": "None",
    },
    "Street_Address": "None",
    "City": "None",
    "State_or_Province": "None",
    "Zip_or_Postal_Code": "None",
    "Country": "None",
    "Phone": "None",
    "Mobile": "None",
    "Fax": "None",
    "Primary_Email": "None",
    "Personal_Email": "None",
    "Education_History": [{"School_Name": "None", "Degree": "None"}],
    "Primary_Language": "None",
    "Languages": [],
    "Education_Level": "None",
    "Diocese": "None",
    "Bishop_Email": "None",
    "Bishop_Name": "None",
    "Seminary_Name": "None",
    "Seminary_Address": "None",
    "Seminary_Email": "None",
    "Religious_Status": "None",
    "Religious_Order": "None",
    "Ordination_Date": "None",
    "Document_Metadata": {
        "Document_Type": "None",
        "Document_Date": "None",
        "Document_Year": "None",
        "Confidence_Scores": {},
    },
}

FINAL_OUTPUT_SCHEMA = json.loads(json.dumps(SCHEMA))
FINAL_OUTPUT_SCHEMA.pop("Document_Metadata", None)
FINAL_OUTPUT_SCHEMA["Confidence"] = {}

# =====================================================
# ZOHO CATALYST & LM STUDIO CONFIGURATION
# =====================================================
VLM_URL = "https://api.catalyst.zoho.com/quickml/v1/project/35939000000182003/vlm/chat"
# VLM_URL = "http://crm-l40s-1.csez.zohocorpin.com:9311/inference"
TOKEN_URL = "https://accounts.zoho.com/oauth/v2/token"
LLM_URL = "https://api.catalyst.zoho.com/quickml/v2/project/35939000000182003/llm/chat"

CATALYST_ORG = "914134238"
MODEL_NAME = "VL-Qwen2.5-7B"
# MODEL_NAME = "VL-Qwen3.6-35B-A3B"
CONSOLIDATION_MODEL_NAME = "crm-di-qwen_text_14b-fp8-it"

CLIENT_ID = "1000.6FIKU7IPS8MCUXWTL1KL0HZRTPS3RH"
CLIENT_SECRET = "30bd0ff9f34a045bf722a5e05ef86cdca404f019bf"
REFRESH_TOKEN = "1000.8fc334b0666d6bb8169c1901490da877.6ff370561224a35c62b23e37da45fd8f"

OCR_MAX_WORKERS = 4         # 8
CONSOLIDATION_MAX_WORKERS = 6
CONSOLIDATION_BATCH_SIZE = 5
FINAL_PROMPT_CHAR_LIMIT = 85000

# Toggle to show/hide the "Total Extraction Time" indicator in the UI.
SHOW_EXTRACTION_TIME = False

# Local folder (inside the project) where uploaded PDFs are persisted.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploaded_files")
os.makedirs(LOCAL_UPLOAD_DIR, exist_ok=True)

