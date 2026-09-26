# config/database.py

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from the .env file in the root directory
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials. Check your .env file.")

# Initialize and export the Supabase client
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)