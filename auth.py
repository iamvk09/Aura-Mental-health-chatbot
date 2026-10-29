"""
User Authentication and Database Management
Handles user registration, login, and encrypted conversation storage
"""

import json
import os
import hashlib
import base64
from cryptography.fernet import Fernet
from datetime import datetime
import re

# Database file
DB_FILE = "data/users.json"
CONVERSATIONS_DIR = "data/conversations"

# Create directories if they don't exist
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

def generate_key_from_password(password):
    """Generate encryption key from password"""
    # Use password to derive a key
    key = hashlib.sha256(password.encode()).digest()
    return base64.urlsafe_b64encode(key)

def load_users():
    """Load users from database"""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to database"""
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_username(username):
    """Validate username (alphanumeric, 3-20 chars)"""
    pattern = r'^[a-zA-Z0-9_]{3,20}$'
    return re.match(pattern, username) is not None

def register_user(username, email, password):
    """Register a new user"""
    users = load_users()
    
    # Check if username exists
    if username in users:
        return False, "Username already exists"
    
    # Check if email exists
    for user_data in users.values():
        if user_data.get('email') == email:
            return False, "Email already registered"
    
    # Validate inputs
    if not validate_username(username):
        return False, "Username must be 3-20 characters, alphanumeric and underscores only"
    
    if not validate_email(email):
        return False, "Invalid email format"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    # Create user
    users[username] = {
        'email': email,
        'password_hash': hash_password(password),
        'created_at': datetime.now().isoformat(),
        'last_login': None
    }
    
    save_users(users)
    return True, "Registration successful"

def login_user(username, password):
    """Authenticate user"""
    users = load_users()
    
    if username not in users:
        return False, "Invalid username or password", None
    
    user_data = users[username]
    
    # Verify password
    if user_data['password_hash'] != hash_password(password):
        return False, "Invalid username or password", None
    
    # Update last login
    user_data['last_login'] = datetime.now().isoformat()
    save_users(users)
    
    return True, "Login successful", {
        'username': username,
        'email': user_data['email']
    }

def encrypt_message(message, password):
    """Encrypt a message using password-derived key"""
    try:
        key = generate_key_from_password(password)
        fernet = Fernet(key)
        encrypted = fernet.encrypt(message.encode())
        return base64.b64encode(encrypted).decode()
    except Exception as e:
        return None

def decrypt_message(encrypted_message, password):
    """Decrypt a message using password-derived key"""
    try:
        key = generate_key_from_password(password)
        fernet = Fernet(key)
        encrypted_bytes = base64.b64decode(encrypted_message.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    except Exception as e:
        return None

def save_conversation(username, user_message, bot_response, password):
    """Save encrypted conversation to history"""
    conversation_file = os.path.join(CONVERSATIONS_DIR, f"{username}.json")
    
    # Load existing conversations
    conversations = []
    if os.path.exists(conversation_file):
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        except:
            conversations = []
    
    # Encrypt messages
    encrypted_user = encrypt_message(user_message, password)
    encrypted_bot = encrypt_message(bot_response, password)
    
    if encrypted_user and encrypted_bot:
        conversations.append({
            'timestamp': datetime.now().isoformat(),
            'user_message_encrypted': encrypted_user,
            'bot_response_encrypted': encrypted_bot
        })
        
        # Save encrypted conversations
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    return len(conversations)

def load_conversation_history(username, password, limit=50):
    """Load and decrypt conversation history for user"""
    conversation_file = os.path.join(CONVERSATIONS_DIR, f"{username}.json")
    
    if not os.path.exists(conversation_file):
        return []
    
    try:
        with open(conversation_file, 'r', encoding='utf-8') as f:
            encrypted_conversations = json.load(f)
        
        # Decrypt and return recent conversations
        history = []
        for conv in encrypted_conversations[-limit:]:
            user_msg = decrypt_message(conv['user_message_encrypted'], password)
            bot_msg = decrypt_message(conv['bot_response_encrypted'], password)
            
            if user_msg and bot_msg:
                history.append({
                    'timestamp': conv['timestamp'],
                    'user_message': user_msg,
                    'bot_response': bot_msg
                })
        
        return history
    except Exception as e:
        return []

