from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from google.cloud import storage
# from google.oauth2 import service_account
# import psycopg2
import tempfile
from psycopg2.extras import RealDictCursor
# from psycopg2.pool import SimpleConnectionPool
import uuid
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Optional
import io
import os
import requests
from db import execute_query, fetch_query_results
from contextlib import contextmanager
import logging
from pydantic import BaseModel
import shutil
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="frontend")
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    './my-poc-94663-service-account.json')

scoped_credentials = credentials.with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform'])
# GCP Configuration
GCP_BUCKET_NAME = "invoice-storage-bucket-aikyro"  # Replace with your bucket name
# For production, use: credentials = service_account.Credentials.from_service_account_file('path/to/key.json')
# For development, ensure GOOGLE_APPLICATION_CREDENTIALS env var is set

# PostgreSQL Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "invoice_management"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password")
}

# Connection pool
db_pool = None

def get_storage_client():
    return storage.Client()


# Predefined categories with keywords
CATEGORIES = {
    "Utilities": ["electricity", "water", "gas", "utility", "power", "energy"],
    "Transportation": ["uber", "lyft", "taxi", "fuel", "gas station", "parking", "toll"],
    "Food & Dining": ["restaurant", "cafe", "food", "dining", "grocery", "supermarket"],
    "Office Supplies": ["office", "stationery", "supplies", "paper", "printer"],
    "Technology": ["software", "hardware", "computer", "laptop", "license", "subscription"],
    "Healthcare": ["medical", "hospital", "pharmacy", "doctor", "health"],
    "Entertainment": ["movie", "theatre", "entertainment", "streaming", "netflix"],
    "Travel": ["hotel", "flight", "airline", "booking", "travel"],
    "Miscellaneous": []
}
@app.get("/")
async def root(request:Request):
    return templates.TemplateResponse("index.html", {"request": {}})
def categorize_invoice(text: str) -> str:
    """Automatically categorize invoice based on content"""
    text_lower = text.lower()
    
    for category, keywords in CATEGORIES.items():
        if category == "Miscellaneous":
            continue
        for keyword in keywords:
            if keyword in text_lower:
                return category
    
    return "Miscellaneous"

def extract_amount(text: str) -> Optional[float]:
    """Extract amount from invoice text"""
    patterns = [
        # Look for keywords like 'total', 'amount', 'balance' followed by a number.
        # This pattern is more flexible with currency symbols and number formats.
        r'(?:total|amount|balance|subtotal)[\s:]*[$₹]?\s*([\d,]+\.\d{2})',
        # A more generic pattern to find monetary values, often the largest on the invoice.
        r'[$₹]?\s*(\d{1,3}(?:,\d{2,3})*\.\d{2})'
    ]
    
    found_amounts = []
    for pattern in patterns:
        # Find all occurrences matching the pattern
        matches = re.findall(pattern, text.lower())
        for amount_str in matches:
            # Clean and convert to float
            cleaned_amount = float(amount_str.replace(',', ''))
            found_amounts.append(cleaned_amount)
    # Return the largest amount found, as it's often the total.
    return max(found_amounts) if found_amounts else None

from pypdf import PdfReader
def extract_invoice_using_pypdf(file_path:str):
    try:
        reader = PdfReader(file_path)
        number_of_pages = len(reader.pages)
        text = ""
        print("Number of pages in PDF:", number_of_pages)
        for i in range(number_of_pages):
            page = reader.pages[i]
            text += page.extract_text()
        print("Extracted Text:", text)  # Debugging line
        return text
    except Exception as e:
        logger.error(f"Error extracting resume using pypdf: {e}")
        raise ValueError(f"An error occurred while extracting resume data using pypdf: {str(e)}")
def sanitize_filename(filename: str) -> str:
    """Removes potentially harmful characters from filename."""
    return "".join(c for c in filename if c.isalnum() or c in ['.', '_', '-'])
class ExtractionError(Exception):
    """Base class for extraction errors."""
    pass
class FileHandlingError(ExtractionError):
    """Error during file processing."""
    pass

@app.post("/api/upload-invoice")
async def upload_invoice(
    file: UploadFile = File(...),
    user_name: str = Form(...)
):
    try:
        # Read file content
        content = await file.read()
        
        # Generate unique ID and storage key
        invoice_id = str(uuid.uuid4())
        storage_key = f"invoices/{user_name}/{invoice_id}/{file.filename}"
        
        # Upload to GCP Cloud Storage
        storage_client = get_storage_client()
        bucket = storage_client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(storage_key)
        blob.upload_from_string(content, content_type=file.content_type)
        sanitized_filename = sanitize_filename(file.filename)
        if sanitized_filename != file.filename:
            logging.warning(f"Filename sanitized from '{file.filename}' to '{sanitized_filename}'")

        # File Upload Validation and Handling
        if file.content_type != "application/pdf":
            raise FileHandlingError("Only PDF files are supported.")
        # Extract text (simple version - use Google Vision API for production)
        tmp_path = None
        extracted_data  = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                await file.seek(0)  # Reset file pointer to the beginning
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
                extracted_data = extract_invoice_using_pypdf(tmp_path)
                print("Extracted Data:", extracted_data)  # Debugging line
        except Exception as e:
            raise FileHandlingError(f"Error saving uploaded file: {e}")
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        # Auto-categorize
        category = categorize_invoice(extracted_data)
        amount = extract_amount(extracted_data) or 0.0
        
        # Store metadata in database
        await execute_query(
            "INSERT INTO users (name, created_at) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
            [user_name]
        )
        
        # Insert invoice record
        await execute_query("""
            INSERT INTO invoices 
            (id, user_name, file_name, storage_key, category, amount, upload_date, extracted_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, [
            invoice_id, user_name, file.filename, storage_key, category, 
            amount, datetime.now(), extracted_data]
        )
        
        return JSONResponse({
            "success": True,
            "invoice_id": invoice_id,
            "category": category,
            "amount": amount,
            "file_name": file.filename,
            "storage_key": storage_key,
            "extracted_text": extracted_data,
            "message": "Invoice uploaded and categorized successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/invoices/{user_name}")
async def get_user_invoices(user_name: str):
    try:
        invoices = await fetch_query_results("""
            SELECT id, file_name, category, amount, upload_date, storage_key
            FROM invoices 
            WHERE user_name = %s 
            ORDER BY upload_date DESC
        """, [user_name])
        print(f"Fetched invoices for user {user_name}: {invoices}")
        return {"invoices": invoices}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/categories")
async def get_categories():
    return JSONResponse({"categories": list(CATEGORIES.keys())})

class InvoiceUpdateRequest(BaseModel):
    user_name: str
    file_name: str
    storage_key: str
    category: str
    amount: float
    extracted_text: Optional[str] = None

@app.post("/api/invoices")
async def save_invoice_details(invoice_data: InvoiceUpdateRequest):
    """
    Updates an invoice with user-confirmed details and marks it as 'confirmed'.
    """
    try:
        # The storage_key is unique per upload, so we use it to find the correct invoice.
        result = await execute_query("""
            UPDATE invoices 
            SET category = %s, amount = %s, extracted_text = %s
            WHERE storage_key = %s AND user_name ~~* %s
            RETURNING id;
        """, [
            invoice_data.category, invoice_data.amount, invoice_data.extracted_text,
            invoice_data.storage_key, f"%{invoice_data.user_name}%"
        ])
        print(f"Update result: {result}")
        return JSONResponse({"success": True, "message": "Invoice saved successfully."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/invoice/{invoice_id}/category")
async def update_category(invoice_id: str, category: str = Form(...)):
    try:
        await execute_query(
            "UPDATE invoices SET category = %s WHERE id = %s",
            [category, invoice_id]
        )
        
        return JSONResponse({"success": True, "message": "Category updated"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/{user_name}")
async def get_user_stats(user_name: str):
    try:

        # Category breakdown
        category_stats  = await fetch_query_results("""
            SELECT category, COUNT(*) as count, COALESCE(SUM(amount::numeric), 0) as total
            FROM invoices
            WHERE user_name = %s
            GROUP BY category
        """, [user_name])
        
        # Total stats
        total_stats_list = await fetch_query_results("""
            SELECT COUNT(*) as count, COALESCE(SUM(amount::numeric), 0) as total
            FROM invoices
            WHERE user_name = %s
        """, (user_name,))
        total_row = total_stats_list[0] if total_stats_list else {'count': 0, 'total': 0}
        
        return {
            "total_invoices": int(total_row.get('count', 0)),
            "total_amount": round(float(total_row.get('total', 0)), 2),
            "category_breakdown": category_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_download_signed_url_v4(bucket_name, blob_name):
    """Generates a v4 signed URL for downloading a blob.

    Note that this method requires a service account key file.
    """
    # bucket_name = 'your-bucket-name'
    # blob_name = 'your-object-name'

    storage_client = storage.Client(credentials=scoped_credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=datetime.utcnow() + timedelta(minutes=15),
        # Allow GET requests using this URL.
        method="GET",
    )
    return url

@app.get("/api/invoice/{invoice_id}/download")
async def download_invoice(invoice_id: str):
    try:
        # Fetch invoice details from the database
        invoice_data = await fetch_query_results(
            "SELECT storage_key, file_name FROM invoices WHERE id = %s",
            [invoice_id]
        )
        if not invoice_data:
            raise HTTPException(status_code=404, detail="Invoice not found")

        invoice = invoice_data[0]
        storage_key = invoice['storage_key']
        file_name = invoice['file_name']
        object_signed_url = generate_download_signed_url_v4(GCP_BUCKET_NAME, storage_key)

        return JSONResponse({
            "download_url": object_signed_url,
            "file_name": file_name
        })
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error downloading invoice {invoice_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not download the invoice.")

@app.get("/api/users")
async def get_users():
    # conn = None
    try:
        users_data = await fetch_query_results("SELECT name FROM users ORDER BY name")
        users = [row['name'] for row in users_data]
        return JSONResponse({"users": users})
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch users.")


@app.delete("/api/invoice/{invoice_id}")
async def delete_invoice(invoice_id: str):
    try:
        row = await fetch_query_results("SELECT storage_key FROM invoices WHERE id = %s", [invoice_id])
        
        if not row:
            raise HTTPException(status_code=404, detail="Invoice not found")
        
        storage_key = row['storage_key']
        
        # Delete from storage
        storage_client = get_storage_client()
        bucket = storage_client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(storage_key)
        blob.delete()
        
        # Delete from database
        await execute_query("DELETE FROM invoices WHERE id = %s", (invoice_id,))
        
        return JSONResponse({"success": True, "message": "Invoice deleted"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/google-consent-url")
async def google_consent_url():
    try:
        redirect_uri = 'http://127.0.0.1:8000/api/google-callback'
        # Generate Google OAuth2 consent URL
        from google_auth_oauthlib.flow import Flow

        flow = Flow.from_client_secrets_file(
            'client_secret_google.json',
            scopes=['https://www.googleapis.com/auth/cloud-platform', "https://www.googleapis.com/auth/userinfo.profile"],
            redirect_uri=redirect_uri
        )

        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )

        return JSONResponse({"consent_url": authorization_url})
    except Exception as e:
        logger.error(f"Error generating Google consent URL: {e}")
        raise HTTPException(status_code=500, detail="Could not generate consent URL.")
    
from google_auth_oauthlib.flow import Flow
@app.get("/api/google-callback")
async def google_callback(request: Request):
    try:
        
        # print(f"Google Callback Data: {json.dumps(request)}")
        data = request.query_params
        print(f"Google Callback Data:", data)
        redirect_uri = 'http://127.0.0.1:8000/api/google-callback'
        flow = Flow.from_client_secrets_file(
            'client_secret_google.json',
            scopes=['https://www.googleapis.com/auth/cloud-platform', "https://www.googleapis.com/auth/userinfo.profile"],
            state=data.get('state'))
        flow.redirect_uri = redirect_uri

        authorization_response = str(request.url)
        flow.fetch_token(authorization_response=authorization_response)

        # Store the credentials in browser session storage, but for security: client_id, client_secret,
        # and token_uri are instead stored only on the backend server.
        credentials = flow.credentials

        # Use the access token to get user info
        userinfo_response = requests.get(
            "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
            headers={"Authorization": f"Bearer {credentials.token}"}
        )
        userinfo_response.raise_for_status()  # Raise an exception for bad status codes
        user_info = userinfo_response.json()

        # Upsert user data into the database
        # This will insert a new user or update an existing one based on the email.
        await execute_query(
            """
            INSERT INTO users (id,google_id, email, name, given_name, family_name, picture_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
            """,
            (str(uuid.uuid4()),user_info.get('id'), user_info.get('email'), user_info.get('name'), user_info.get('given_name'), user_info.get('family_name'), user_info.get('picture'))
        )
        response = RedirectResponse(url="/")
        response.set_cookie(key="invoice_user", value=user_info.get('name'), httponly=False, max_age=3600, samesite='lax')
        return response
    except Exception as e:
        logger.error(f"Error in Google callback: {e}")
        raise HTTPException(status_code=500, detail="Error processing callback.")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)