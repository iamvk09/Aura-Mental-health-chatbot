"""
Web-based Chatbot Interface using Flask
Offline RAG-based Mental Health Chatbot
"""

from flask import Flask, render_template, request, jsonify, session
from sentence_transformers import SentenceTransformer
from functools import wraps

# Import chatbot module and constants
import chat
from chat import (
    MODEL_NAME,
    INDEX_DIR,
    TOP_K,
    load_index_and_metadata,
    detect_crisis,
    get_crisis_response,
    detect_greeting,
    get_greeting_response,
    detect_thanks,
    get_thanks_response,
    search_similar_chunks,
    generate_response
)

# Import authentication and topic detection
import auth
import topic_detector
import assessment

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this in production!

# Global variables for model and index
model = None
index = None
metadata_dict = None
metadata_list = None
sources = None

def initialize_chatbot():
    """Initialize the chatbot model and index"""
    global model, index, metadata_dict, metadata_list, sources
    
    if model is None:
        print("Loading model...")
        model = SentenceTransformer(MODEL_NAME)
        print(f"Model {MODEL_NAME} loaded")
    
    if index is None:
        print("Loading index...")
        index, metadata_dict = load_index_and_metadata(INDEX_DIR)
        metadata_list = metadata_dict.get('metadata_list', [])
        sources = metadata_dict.get('sources', [])
        print("Index loaded successfully")

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index_page():
    """Render the main page (login or chat)"""
    if 'username' in session:
        return render_template('chat.html', username=session.get('username'))
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    """Handle user registration"""
    data = request.json
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    
    success, message = auth.register_user(username, email, password)
    
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'message': message}), 400

@app.route('/login', methods=['POST'])
def login():
    """Handle user login"""
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    success, message, user_data = auth.login_user(username, password)
    
    if success:
        session['username'] = username
        session['password'] = password  # Store for encryption (in production, use better method)
        # Initialize assessment state
        session['assessment_complete'] = False
        session['assessment_answers'] = []
        session['current_question'] = 1
        return jsonify({
            'success': True, 
            'message': message,
            'username': username
        })
    else:
        return jsonify({'success': False, 'message': message}), 401

@app.route('/logout', methods=['POST'])
def logout():
    """Handle user logout"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/assessment/status', methods=['GET'])
@login_required
def get_assessment_status():
    """Get assessment status"""
    assessment_complete = session.get('assessment_complete', False)
    current_question = session.get('current_question', 1)
    answers = session.get('assessment_answers', [])
    
    return jsonify({
        'complete': assessment_complete,
        'current_question': current_question,
        'total_questions': 5,
        'answers_count': len(answers)
    })

@app.route('/assessment/question', methods=['GET'])
@login_required
def get_current_question():
    """Get current assessment question"""
    current_question_id = session.get('current_question', 1)
    question = assessment.get_question_by_id(current_question_id)
    
    if question:
        return jsonify({
            'question_id': question['id'],
            'question': question['question'],
            'options': question['options']
        })
    return jsonify({'error': 'No more questions'}), 404

@app.route('/assessment/answer', methods=['POST'])
@login_required
def submit_answer():
    """Submit assessment answer"""
    data = request.json
    answer = data.get('answer', '').strip()
    
    if not answer:
        return jsonify({'error': 'Answer required'}), 400
    
    current_question_id = session.get('current_question', 1)
    answers = session.get('assessment_answers', [])
    
    # Add answer
    answers.append(answer)
    session['assessment_answers'] = answers
    
    # Move to next question
    if current_question_id < 5:
        session['current_question'] = current_question_id + 1
        return jsonify({
            'success': True,
            'next_question': current_question_id + 1,
            'complete': False
        })
    else:
        # Assessment complete
        session['assessment_complete'] = True
        score = assessment.calculate_assessment_score(answers)
        feedback = assessment.get_assessment_feedback(score)
        
        return jsonify({
            'success': True,
            'complete': True,
            'feedback': feedback,
            'score': score
        })

@app.route('/history', methods=['GET'])
@login_required
def get_history():
    """Get conversation history for logged-in user"""
    username = session.get('username')
    password = session.get('password')
    
    if not password:
        return jsonify({'history': [], 'username': username})
    
    history = auth.load_conversation_history(username, password, limit=50)
    return jsonify({
        'history': history, 
        'username': username,
        'assessment_complete': session.get('assessment_complete', False)
    })

@app.route('/chat', methods=['POST'])
@login_required
def chat_endpoint():
    """Handle chat messages"""
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        username = session.get('username')
        password = session.get('password')
        
        if not user_input:
            return jsonify({'response': 'Please enter a message.'})
        
        # Check if assessment is complete
        assessment_complete = session.get('assessment_complete', False)
        if not assessment_complete:
            return jsonify({
                'error': 'assessment_required',
                'message': 'Please complete the mental health assessment first.'
            }), 400
        
        # Check if query is mental health related
        is_mental_health, topic = topic_detector.detect_mental_health_topic(user_input)
        if not is_mental_health:
            redirect_msg = topic_detector.get_redirect_message(user_input, topic)
            return jsonify({'response': redirect_msg, 'type': 'redirect'})
        
        # Initialize if needed
        initialize_chatbot()
        
        # Check for crisis keywords
        if detect_crisis(user_input):
            response = get_crisis_response()
            # Save conversation
            if password:
                auth.save_conversation(username, user_input, response, password)
            return jsonify({'response': response, 'type': 'crisis'})
        
        # Check for greetings - personalize with username
        if detect_greeting(user_input):
            response = get_greeting_response(username)
            # Save conversation
            if password:
                auth.save_conversation(username, user_input, response, password)
            return jsonify({'response': response, 'type': 'greeting'})
        
        # Check for thank you - provide improvement suggestions
        if detect_thanks(user_input):
            suggestions = assessment.get_improvement_suggestions()
            response = get_thanks_response() + "\n\nHere are 5 ways to improve your mental health:\n\n"
            for i, suggestion in enumerate(suggestions, 1):
                response += f"{i}. {suggestion}\n"
            # Save conversation
            if password:
                auth.save_conversation(username, user_input, response, password)
            return jsonify({'response': response, 'type': 'thanks'})
        
        # RAG Retrieval: Search for similar chunks
        distances, indices, query_embedding = search_similar_chunks(
            user_input, index, model, TOP_K
        )
        
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
        
        # Save conversation (encrypted)
        if password:
            auth.save_conversation(username, user_input, response, password)
        
        return jsonify({'response': response, 'type': 'normal'})
        
    except Exception as e:
        return jsonify({'response': f"I'm sorry, I encountered an error: {str(e)}. Please try again."}), 500

@app.route('/profile')
@login_required
def profile():
    """User profile page with chat history"""
    return render_template('profile.html', username=session.get('username'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/sos', methods=['POST'])
@login_required
def sos():
    """SOS emergency response"""
    username = session.get('username')
    password = session.get('password')
    
    sos_message = "User activated SOS button"
    sos_response = get_crisis_response()
    
    # Save SOS interaction to history
    if password:
        auth.save_conversation(username, sos_message, sos_response, password)
    
    return jsonify({
        'message': sos_response,
        'emergency_numbers': {
            'india_aasra': '9152987821',
            'india_vandrevala': '1860-2662-345',
            'international': '988'  # International suicide prevention
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Web-based Mental Health Chatbot")
    print("=" * 60)
    print("Initializing chatbot...")
    initialize_chatbot()
    print("\n" + "=" * 60)
    print("Chatbot ready!")
    print("Open your browser and go to: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

