"""
Offline RAG-based Chatbot with NLP
Uses Retrieval-Augmented Generation (RAG) pipeline with:
- Semantic search using FAISS
- NLP-based context understanding
- Intelligent response synthesis
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
import math

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "data/chatbot_index"
TOP_K = 5

# Crisis detection keywords
CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end my life', 'want to die', 'not worth living',
    'self harm', 'hurt myself', 'cut myself', 'self-harm', 'suicidal',
    'taking my life', 'ending it all', 'no point living'
]

# Greeting patterns
GREETING_PATTERNS = [
    r'\b(hi|hello|hey|hii|howdy)\b',
    r'\bgood\s+(morning|afternoon|evening|day)\b',
    r'\bhow\s+are\s+you\b',
    r'\bwhat\'?s\s+up\b',
    r'\bis\s+anyone\s+there\b',
    r'\bgood\s+to\s+see\s+you\b'
]

# Thank you patterns
THANK_PATTERNS = [
    r'\b(thank|thanks|thankful|grateful|appreciate)\b',
    r'\bthanks?\s+(you|a\s+lot|very\s+much|so\s+much)\b',
    r'\bi\s+(am\s+)?grateful\b',
    r'\bi\s+appreciate\b'
]

# Treatment/cure request patterns - only provide treatment advice if these are present
TREATMENT_REQUEST_PATTERNS = [
    r'\b(how\s+do\s+I|how\s+can\s+I|how\s+to)\s+(improve|cure|treat|manage|deal\s+with|cope\s+with|handle|overcome|get\s+rid\s+of|fix)',
    r'\b(what\s+can\s+I\s+do|what\s+should\s+I\s+do|ways\s+to|methods\s+to|tips\s+to|steps\s+to)\s+(improve|cure|treat|manage|deal|cope|handle|overcome)',
    r'\b(cure|treatment|treat|manage|improve|cope|handle|overcome|solution|remedy|help\s+with)',
    r'\b(how\s+to\s+get\s+better|how\s+to\s+feel\s+better|how\s+to\s+recover)'
]

# Treatment-related keywords that should be filtered unless treatment is requested
TREATMENT_KEYWORDS = [
    'treatment', 'cure', 'therapy', 'medication', 'medications', 'antidepressant',
    'cbt', 'cognitive behavioral therapy', 'psychotherapy', 'counseling',
    'should try', 'can try', 'practice', 'techniques', 'exercises', 'methods',
    'ways to', 'steps to', 'how to manage', 'how to cope', 'how to deal'
]

def load_index_and_metadata(index_dir):
    """Load FAISS index and metadata"""
    print("Loading chatbot index...")
    
    index_path = os.path.join(index_dir, "faiss_index.bin")
    metadata_path = os.path.join(index_dir, "metadata.json")
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index not found: {index_path}\n"
            "Please run ingest.py first to create the index."
        )
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}\n"
            "Please run ingest.py first to create the metadata."
        )
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_dict = json.load(f)
    
    print(f"Loaded index with {index.ntotal} vectors")
    return index, metadata_dict

def detect_crisis(user_input):
    """Detect if user input contains crisis keywords"""
    user_lower = user_input.lower()
    for keyword in CRISIS_KEYWORDS:
        if keyword in user_lower:
            return True
    return False

def detect_greeting(user_input):
    """Detect if user input is a greeting"""
    user_lower = user_input.lower().strip()
    
    # Check if it's primarily a greeting (short messages are likely greetings)
    if len(user_input.split()) <= 4:
        for pattern in GREETING_PATTERNS:
            if re.search(pattern, user_lower):
                return True
    
    # Check for greeting-only messages
    greeting_only = True
    for word in user_lower.split():
        if word not in ['hi', 'hello', 'hey', 'hii', 'howdy', 'good', 'morning', 'afternoon', 
                       'evening', 'day', 'how', 'are', 'you', 'what', 's', 'up', 'is', 
                       'anyone', 'there', 'to', 'see', 'chatbot', 'bot']:
            greeting_only = False
            break
    
    return greeting_only and len(user_lower.split()) <= 6

def detect_thanks(user_input):
    """Detect if user input contains thank you expressions"""
    user_lower = user_input.lower()
    for pattern in THANK_PATTERNS:
        if re.search(pattern, user_lower):
            return True
    return False

def detect_treatment_request(user_input):
    """Detect if user is explicitly asking for treatment/cure advice"""
    user_lower = user_input.lower()
    for pattern in TREATMENT_REQUEST_PATTERNS:
        if re.search(pattern, user_lower):
            return True
    return False

def filter_treatment_content(sentences, allow_treatment=False):
    """Filter out treatment advice unless explicitly requested"""
    if allow_treatment:
        return sentences  # Return all sentences if treatment is requested
    
    # Filter out sentences containing treatment keywords
    filtered = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Check if sentence contains treatment keywords
        has_treatment = any(keyword in sentence_lower for keyword in TREATMENT_KEYWORDS)
        
        # Keep informational sentences, filter treatment advice
        if not has_treatment:
            filtered.append(sentence)
        # But keep sentences that are just informational (not prescriptive)
        elif any(word in sentence_lower for word in ['is', 'are', 'refers to', 'means', 'definition']):
            # Keep definitional/informational content
            filtered.append(sentence)
    
    return filtered if filtered else sentences  # Return original if all filtered out

def get_greeting_response(username=None):
    """Return appropriate greeting response"""
    import random
    if username:
        greetings = [
            f"Hello {username}! I'm here to support you. How are you feeling today?",
            f"Hi {username}! I'm glad you're here. What's on your mind?",
            f"Hello {username}! Thank you for reaching out. How can I help you today?",
            f"Hi {username}! I'm here to listen and support you. What would you like to talk about?",
            f"Hello there {username}! I'm glad you're here. How are you doing today?",
            f"Hi {username}! It's good to see you. What brings you here today?",
            f"Hello {username}! I'm here to help. How are you feeling?",
        ]
    else:
        greetings = [
            "Hello! I'm here to support you. How are you feeling today?",
            "Hi there! I'm glad you're here. What's on your mind?",
            "Hello! Thank you for reaching out. How can I help you today?",
            "Hi! I'm here to listen and support you. What would you like to talk about?",
            "Hello there! I'm glad you're here. How are you doing today?",
            "Hi! It's good to see you. What brings you here today?",
            "Hello! I'm here to help. How are you feeling?",
        ]
    return random.choice(greetings)

def get_thanks_response():
    """Return appropriate thank you response"""
    import random
    thanks_responses = [
        "You're very welcome! I'm glad I could help. Feel free to reach out anytime you need support.",
        "You're welcome! I'm here for you whenever you need someone to talk to.",
        "It's my pleasure to help! Remember, I'm always here when you need support.",
        "You're so welcome! Take care of yourself, and don't hesitate to come back if you need anything.",
        "I'm happy I could help! Your well-being matters, so please reach out anytime.",
        "You're welcome! I'm here to support you. Take care!",
    ]
    return random.choice(thanks_responses)

def get_crisis_response():
    """Return crisis response message"""
    return (
        "I'm really sorry you're feeling this way.\n\n"
        "Your safety matters. I cannot provide crisis help.\n\n"
        "Please contact your local emergency number or a suicide prevention hotline immediately.\n\n"
        "In India: Call Aasra at 9152987821 (24x7)."
    )

def preprocess_text(text):
    """NLP preprocessing: clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'"]', '', text)
    return text

def extract_key_phrases(text, max_phrases=5):
    """Extract key phrases from text using simple NLP"""
    # Simple key phrase extraction based on word frequency
    words = re.findall(r'\b\w+\b', text.lower())
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
    
    filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
    word_freq = Counter(filtered_words)
    return [word for word, _ in word_freq.most_common(max_phrases)]

def calculate_relevance_score(query_embedding, chunk_embedding, distance):
    """Calculate relevance score combining semantic similarity and distance"""
    # Convert distance to similarity (for cosine similarity, higher is better)
    similarity = float(distance)
    # Normalize to 0-1 range (assuming normalized embeddings)
    relevance = max(0, min(1, similarity))
    return relevance

def search_similar_chunks(query, index, model, top_k=5):
    """RAG Retrieval: Search for similar chunks using FAISS with NLP-enhanced query"""
    # Preprocess query
    processed_query = preprocess_text(query)
    
    # Generate query embedding
    query_embedding = model.encode(
        [processed_query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype('float32')
    
    # Search in FAISS index
    distances, indices = index.search(query_embedding, top_k)
    
    return distances[0], indices[0], query_embedding[0]

def format_sources(sources, unique=True):
    """Format sources list for display"""
    if unique:
        # Get unique sources
        seen = set()
        unique_sources = []
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
        sources = unique_sources
    
    formatted = []
    for i, source in enumerate(sources, 1):
        formatted.append(f"{i}. {source}")
    return "\n".join(formatted)

def rank_chunks_by_relevance(chunks_data, query_embedding, model):
    """NLP-based ranking: Rank chunks by relevance to query"""
    ranked_chunks = []
    
    for chunk_data in chunks_data:
        idx = chunk_data['index']
        text = chunk_data['text']
        distance = chunk_data['distance']
        
        # Generate embedding for chunk text to calculate similarity
        chunk_embedding = model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].astype('float32')
        
        # Calculate relevance score
        relevance = calculate_relevance_score(query_embedding, chunk_embedding, distance)
        
        # Extract key phrases for additional matching
        key_phrases = extract_key_phrases(text)
        
        ranked_chunks.append({
            **chunk_data,
            'relevance': relevance,
            'key_phrases': key_phrases
        })
    
    # Sort by relevance (highest first)
    ranked_chunks.sort(key=lambda x: x['relevance'], reverse=True)
    return ranked_chunks

def synthesize_context(ranked_chunks, query="", max_chunks=3):
    """RAG Synthesis: Intelligently combine multiple chunks into coherent context"""
    if not ranked_chunks:
        return ""
    
    # Select top chunks based on relevance
    top_chunks = ranked_chunks[:max_chunks]
    
    # Extract query keywords for better matching
    query_keywords = set(extract_key_phrases(query.lower(), max_phrases=15)) if query else set()
    
    # Combine chunks, removing redundancy and prioritizing relevant content
    context_sentences = []
    seen_phrases = set()
    sentence_scores = []
    
    for chunk in top_chunks:
        text = chunk['text']
        relevance = chunk.get('relevance', 0.5)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue
            
            # Calculate sentence relevance score
            sentence_keywords = set(extract_key_phrases(sentence.lower(), max_phrases=10))
            keyword_overlap = len(query_keywords.intersection(sentence_keywords))
            score = relevance * 0.5 + (keyword_overlap / max(len(query_keywords), 1)) * 0.5
            
            # Check for redundancy using key phrases
            sentence_phrases = set(extract_key_phrases(sentence, max_phrases=3))
            is_redundant = sentence_phrases.issubset(seen_phrases) if sentence_phrases else False
            
            if not is_redundant:
                sentence_scores.append((score, sentence, sentence_keywords))
                seen_phrases.update(sentence_phrases)
    
    # Sort by relevance score and take top sentences
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    context_sentences = [s for _, s, _ in sentence_scores[:6]]  # Top 6 most relevant
    
    if context_sentences:
        return ". ".join(context_sentences) + "."
    return ""

def generate_response(query, retrieved_chunks, distances, sources, metadata_list, query_embedding, model):
    """RAG Generation: Generate empathetic response using retrieved context"""
    # Prepare chunk data with distances
    chunks_data = []
    retrieved_sources = []
    
    for i, idx in enumerate(retrieved_chunks):
        if idx < len(metadata_list):
            metadata = metadata_list[idx]
            source = sources[idx] if idx < len(sources) else metadata.get('source', f"Source {idx + 1}")
            text = metadata.get('text', '')
            
            if text:
                chunks_data.append({
                    'index': idx,
                    'text': text,
                    'source': source,
                    'distance': float(distances[i]) if i < len(distances) else 0.0,
                    'metadata': metadata
                })
                retrieved_sources.append(source)
    
    if not chunks_data:
        return "I'm sorry, I couldn't find relevant information to help with that. Could you try rephrasing your question?"
    
    # Rank chunks by relevance using NLP
    ranked_chunks = rank_chunks_by_relevance(chunks_data, query_embedding, model)
    
    # Synthesize context from ranked chunks (pass query for better matching)
    context = synthesize_context(ranked_chunks, query=query, max_chunks=3)
    
    # Build response - more natural flow
    response_parts = []
    
    # Create empathetic response from synthesized context (main response)
    supportive_response = create_empathetic_response(query, context, ranked_chunks)
    response_parts.append(supportive_response)
    
    # No sources added - removed as requested
    
    return "\n".join(response_parts)

def extract_query_intent(query):
    """NLP: Extract the intent/question from user query"""
    # Remove greeting words to get the actual question
    query_lower = query.lower()
    greeting_words = ['hi', 'hello', 'hey', 'hii', 'howdy', 'good morning', 'good afternoon', 
                     'good evening', 'thanks', 'thank you']
    
    for word in greeting_words:
        query_lower = query_lower.replace(word, '').strip()
    
    # Clean up
    query_lower = re.sub(r'\s+', ' ', query_lower).strip()
    return query_lower if query_lower else query.lower()

def find_most_relevant_sentences(query, sentences, ranked_chunks, max_sentences=3):
    """NLP: Find sentences most relevant to the query"""
    if not sentences:
        return []
    
    query_words = set(extract_key_phrases(query, max_phrases=10))
    sentence_scores = []
    
    for sentence in sentences:
        sentence_words = set(extract_key_phrases(sentence, max_phrases=10))
        # Calculate overlap
        overlap = len(query_words.intersection(sentence_words))
        # Boost score if sentence contains query words
        score = overlap + (1 if any(word in sentence.lower() for word in query.lower().split() if len(word) > 3) else 0)
        sentence_scores.append((score, sentence))
    
    # Sort by score and return top sentences
    sentence_scores.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in sentence_scores[:max_sentences]]

def create_empathetic_response(query, context, ranked_chunks):
    """RAG-based Response Generation: Create empathetic, specific, context-aware response"""
    import random
    
    if not context or len(context.strip()) < 50:
        return create_fallback_response(query)
    
    # Check if treatment advice is requested
    treatment_requested = detect_treatment_request(query)
    
    # NLP: Extract key information from context
    sentences = [s.strip() for s in re.split(r'[.!?]+', context) if s.strip() and len(s.strip()) > 20]
    
    if not sentences:
        return create_fallback_response(query)
    
    # Filter treatment content unless explicitly requested
    sentences = filter_treatment_content(sentences, allow_treatment=treatment_requested)
    
    if not sentences:
        # If all filtered out and treatment not requested, provide informational response
        if not treatment_requested:
            return "I understand you're asking about this topic. For specific treatment or management advice, please ask explicitly, for example: 'How can I manage this?' or 'What can I do to improve?'"
        return create_fallback_response(query)
    
    # Extract query intent for better matching
    query_intent = extract_query_intent(query)
    
    # Find most relevant sentences to the query
    relevant_sentences = find_most_relevant_sentences(query_intent, sentences, ranked_chunks, max_sentences=3)
    
    # If no highly relevant sentences, use top sentences from context
    if not relevant_sentences:
        relevant_sentences = sentences[:min(3, len(sentences))]
    
    # Build response using RAG principles
    response_parts = []
    
    # Define query_lower for use in the function
    query_lower = query.lower()
    
    # Core information - be specific and direct (no opening phrases)
    if relevant_sentences:
        # Use sentences directly without excessive connectors
        core_info = " ".join(relevant_sentences)
        # Ensure proper punctuation
        if not core_info.rstrip().endswith(('.', '!', '?')):
            core_info += "."
        
        response_parts.append(core_info)
    
    # Only provide actionable advice if treatment is requested (as bullet points)
    if treatment_requested:
        action_keywords = ['can', 'should', 'try', 'practice', 'seek', 'help', 'support', 'important', 'effective', 'techniques', 'methods']
        actionable_sentences = [s for s in sentences if any(kw in s.lower() for kw in action_keywords)]
        
        # Filter actionable sentences that weren't already used
        new_actionable = [s for s in actionable_sentences if s not in relevant_sentences]
        
        if new_actionable:
            # Format as bullet points on separate lines
            for actionable in new_actionable[:5]:  # Max 5 items
                # Clean and format the actionable item
                actionable_clean = actionable.strip()
                # Remove leading connectors and make it direct
                actionable_clean = re.sub(r'^(Additionally|Furthermore|Also|Moreover|Similarly|For example|Such as)[,\s]*', '', actionable_clean, flags=re.IGNORECASE)
                actionable_clean = actionable_clean.strip()
                
                # Ensure it starts with a verb or "You can"
                if not actionable_clean.lower().startswith(('you', 'try', 'practice', 'seek', 'engage', 'consider', 'use', 'apply', 'develop', 'build', 'maintain', 'create', 'establish')):
                    # Try to extract the actionable part
                    actionable_clean = actionable_clean
                
                # Add bullet point (join will add newline)
                response_parts.append(f"• {actionable_clean}")
    
    # Brief, supportive closing (only if not treatment request)
    if not treatment_requested:
        if any(word in query_lower for word in ['anxious', 'worried', 'stressed', 'depressed', 'sad']):
            response_parts.append("If you'd like specific strategies to manage this, feel free to ask.")
        else:
            response_parts.append("If you need more specific information, please let me know.")
    
    return "\n".join(response_parts)

def create_fallback_response(query):
    """Fallback response when context is insufficient"""
    import random
    
    fallback_responses = [
        "I understand you're looking for support. While I don't have specific information about that right now, I want you to know that your feelings are valid and important. Consider reaching out to a mental health professional who can provide personalized guidance.",
        "Thank you for sharing that with me. I'm here to listen and support you. For more specific help, I'd recommend speaking with a mental health professional who can provide tailored support for your situation.",
    ]
    
    return random.choice(fallback_responses)

def main():
    """Main chat loop"""
    print("=" * 60)
    print("Offline RAG-based Mental Health Chatbot")
    print("Powered by NLP & Retrieval-Augmented Generation")
    print("=" * 60)
    print("Loading model and index...")
    
    # Load model
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model {MODEL_NAME} loaded")
    
    # Load index and metadata
    index, metadata_dict = load_index_and_metadata(INDEX_DIR)
    
    metadata_list = metadata_dict.get('metadata_list', [])
    sources = metadata_dict.get('sources', [])
    
    print("\n" + "=" * 60)
    print("Chatbot ready! Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("=" * 60)
    print()
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nBot: Take care, and remember you're not alone. Be well! 🌟")
                break
            
            if not user_input:
                continue
            
            # Check for crisis keywords
            if detect_crisis(user_input):
                print("\nBot:")
                print(get_crisis_response())
                print()
                continue
            
            # Check for greetings
            if detect_greeting(user_input):
                print("\nBot:")
                print(get_greeting_response())
                print()
                continue
            
            # Check for thank you
            if detect_thanks(user_input):
                print("\nBot:")
                print(get_thanks_response())
                print()
                continue
            
            # RAG Retrieval: Search for similar chunks
            distances, indices, query_embedding = search_similar_chunks(user_input, index, model, TOP_K)
            
            # Get retrieved chunks metadata
            retrieved_chunks = indices.tolist()
            
            # RAG Generation: Generate response using retrieved context
            response = generate_response(
                user_input, 
                retrieved_chunks, 
                distances,
                sources, 
                metadata_list,
                query_embedding,
                model
            )
            
            # Display response
            print("\nBot:")
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nBot: Take care! Be well! 🌟")
            break
        except Exception as e:
            print(f"\nBot: I'm sorry, I encountered an error: {str(e)}")
            print("Please try again or rephrase your question.\n")

if __name__ == "__main__":
    main()

