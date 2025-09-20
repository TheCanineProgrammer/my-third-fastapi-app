from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from datasets import load_dataset
import os
import openai
import pandas as pd
from rapidfuzz import process, fuzz
import logging

# ------------------------------
# Logging configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ------------------------------
# OpenAI API setup
# ------------------------------
#openai.api_key = "trb-efb1e474594463bbea-27e8-4578-b6ff-b2aed5414b57"
#openai.api_base = "https://turbo.torob.com/v1"

# ------------------------------
# FastAPI setup
# ------------------------------
app = FastAPI()

# ------------------------------
# Pydantic models
# ------------------------------
class Message(BaseModel):
    type: str
    content: str

class ChatRequest(BaseModel):
    chat_id: str
    messages: List[Message]

# ------------------------------
# Load dataset
# ------------------------------

dataset = load_dataset("The-CaPr-2025/base_products")  # pulls from HF Hub
base_df = dataset["train"].to_pandas()

#random_key_to_names = base_df.set_index("random_key")[["english_name", "persian_name"]].to_dict(orient="index")
all_names = list(base_df.persian_name) + list(base_df.english_name)

# ------------------------------
# Endpoint
# ------------------------------
@app.post("/chat")
async def assistant(request: ChatRequest):
    last_message = request.messages[-1].content.strip()
    client = openai.OpenAI(api_key="trb-efb1e474594463bbea-27e8-4578-b6ff-b2aed5414b57", base_url="https://turbo.torob.com/v1")
    # Log incoming request
    logging.info(f"Received chat_id={request.chat_id}, message={last_message}")

    # --------
    # Scenario 1: Sanity checks
    # --------
    if last_message.lower() == "ping":
        response = {"message": "pong", "base_random_keys": None, "member_random_keys": None}
        logging.info(f"Response: {response}")
        return response

    if "return base random key" in last_message.lower():
        key = last_message.split(":")[-1].strip()
        response = {"message": None, "base_random_keys": [key], "member_random_keys": None}
        logging.info(f"Response: {response}")
        return response

    if "return member random key" in last_message.lower():
        key = last_message.split(":")[-1].strip()
        response = {"message": None, "base_random_keys": None, "member_random_keys": [key]}
        logging.info(f"Response: {response}")
        return response
    
    try:
        intent_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are an intent classifier. Decide if the user is asking about a product's feature (like size, width, weight, etc.) or if they want the product itself."},
                {"role": "user", "content": f"User query: {last_message}\n\nAnswer ONLY with one word: 'feature' or 'product'."}
            ],
            temperature=0.0
        )
        intent = intent_response.choices[0].message.content.strip().lower()
    except Exception as e:
        logging.error(f"OpenAI intent error: {e}")
        intent = "product"

    # --------
    # Scenario 2: Map query to a base product key using OpenAI
    # --------
    prompt = f'''
    کاربر یک محصول را توصیف کرده است.
    لطفاً فقط نام یا توضیح استاندارد همان محصول را بازگردان (مثلاً «فرشینه مخمل ترمزگیر عرض 1 متر طرح آشپزخانه کد 04»).
    
    کوئری کاربر: "{last_message}"'''

    key = None
    try:
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You normalize user product queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        predicted_name = response.choices[0].message.content.strip()
        logging.info(f"{predicted_name}")
    except Exception:
        predicted_name = ""
    
    key = None
    matched_row = None
    if predicted_name:
        match, score, idx = process.extractOne(
            predicted_name,
            all_names,
            scorer=fuzz.token_sort_ratio
        )

        if score > 70:  # accept only strong matches
            matched_row = base_df.iloc[idx]
            key = matched_row["random_key"]
    
    if intent == "feature" and matched_row is not None:
        feature = matched_row["extra_features"]
        prompt1 = f"""
    کاربر درباره‌ی ویژگی خاص یک محصول پرسیده است.
    ویژگی‌های محصول: {feature}

    فقط و فقط جواب مدنظر کاربر را برگردان (به انگلیسی).
    مثال: اگر در ویژگی‌ها نوشته شده "1.18 meter"، فقط همان "1.18 meter" را برگردان.
    """
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You return only the requested feature."},
                    {"role": "user", "content": prompt1}
                ],
                temperature=0.0
            )
            feature_answer = response.choices[0].message.content.strip()
            return {
                "message": feature_answer,
                "base_random_keys": None,
                "member_random_keys": None
            }
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
    else:
        response = {
            "message": None,
            "base_random_keys": [key] if key else None,
            "member_random_keys": None
        }

        # Log outgoing response
        logging.info(f"Response: {response}")
        return response
