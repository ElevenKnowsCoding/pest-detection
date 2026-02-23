from functools import wraps
from flask import session, redirect, url_for, request
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Authentication key - change in .env file
AUTH_KEY = os.environ.get('AUTH_KEY', 'pest2024')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def check_auth(key):
    return key == AUTH_KEY
