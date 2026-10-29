"""
Mental Health Topic Detection
Detects if a query is related to mental health or should be redirected
"""

import re
from sentence_transformers import SentenceTransformer
import numpy as np

# Mental health keywords and topics
MENTAL_HEALTH_KEYWORDS = [
    'mental health', 'anxiety', 'depression', 'stress', 'therapy', 'counseling',
    'psychology', 'psychiatrist', 'psychologist', 'emotion', 'feeling', 'feelings',
    'sad', 'happy', 'angry', 'worried', 'nervous', 'panic', 'fear', 'phobia',
    'trauma', 'ptsd', 'bipolar', 'schizophrenia', 'ocd', 'adhd', 'autism',
    'self-esteem', 'confidence', 'self-care', 'wellbeing', 'well-being',
    'suicide', 'self-harm', 'coping', 'grief', 'loss', 'bereavement',
    'addiction', 'substance', 'alcohol', 'drug', 'recovery', 'rehab',
    'meditation', 'mindfulness', 'relaxation', 'breathing', 'exercise',
    'sleep', 'insomnia', 'eating disorder', 'anorexia', 'bulimia',
    'relationship', 'marriage', 'divorce', 'loneliness', 'isolation',
    'workplace stress', 'burnout', 'work-life balance', 'career stress',
    'academic stress', 'exam anxiety', 'student mental health',
    'elderly mental health', 'aging', 'dementia', 'alzheimer',
    'child mental health', 'adolescent', 'teen', 'youth',
    'postpartum', 'pregnancy', 'maternal mental health',
    'veteran mental health', 'military', 'ptsd',
    'lgbtq mental health', 'gender', 'identity',
    'crisis', 'emergency', 'hotline', 'support', 'help'
]

# Non-mental health topics that should be redirected
GENERAL_TOPICS = [
    'prime minister', 'president', 'politics', 'government', 'election',
    'sports', 'football', 'cricket', 'basketball', 'soccer',
    'weather', 'temperature', 'climate', 'rain', 'snow',
    'cooking', 'recipe', 'food', 'restaurant', 'cuisine',
    'technology', 'computer', 'programming', 'software', 'hardware',
    'science', 'physics', 'chemistry', 'biology', 'mathematics',
    'history', 'geography', 'country', 'city', 'capital',
    'entertainment', 'movie', 'music', 'celebrity', 'actor',
    'shopping', 'price', 'buy', 'purchase', 'product',
    'travel', 'vacation', 'hotel', 'flight', 'tourism'
]

def detect_mental_health_topic(query):
    """Detect if query is related to mental health"""
    query_lower = query.lower()
    
    # Check for mental health keywords
    for keyword in MENTAL_HEALTH_KEYWORDS:
        if keyword in query_lower:
            return True, "mental_health"
    
    # Check for general topics (non-mental health)
    for topic in GENERAL_TOPICS:
        if topic in query_lower:
            return False, topic
    
    # Check for question words that might indicate non-mental health queries
    question_patterns = [
        r'who is (the|a)',
        r'what is (the|a)',
        r'where is (the|a)',
        r'when did',
        r'how many',
        r'how much'
    ]
    
    for pattern in question_patterns:
        if re.search(pattern, query_lower):
            # If it's a factual question without mental health context, likely not mental health
            if not any(keyword in query_lower for keyword in MENTAL_HEALTH_KEYWORDS):
                return False, "general_question"
    
    # Default: assume it's mental health related if unclear
    # This is a mental health chatbot, so we err on the side of being helpful
    return True, "mental_health"

def get_redirect_message(query, detected_topic):
    """Get appropriate redirect message for non-mental health queries"""
    return (
        f"I'm a mental health support chatbot, and I'm here to help with mental health, "
        f"emotional wellbeing, and related topics. I noticed your question about '{detected_topic}' "
        f"isn't related to mental health.\n\n"
        f"Please feel free to ask me about:\n"
        f"• Anxiety, depression, stress, or other mental health concerns\n"
        f"• Coping strategies and self-care techniques\n"
        f"• Understanding mental health conditions\n"
        f"• Seeking help and support resources\n\n"
        f"How can I help you with your mental health today?"
    )

