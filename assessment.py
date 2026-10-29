"""
Mental Health Assessment Module
5 fixed questions to assess user's mental health
"""

# Fixed assessment questions
ASSESSMENT_QUESTIONS = [
    {
        'id': 1,
        'question': 'How would you rate your overall mood over the past week?',
        'options': ['Very positive', 'Somewhat positive', 'Neutral', 'Somewhat negative', 'Very negative']
    },
    {
        'id': 2,
        'question': 'How often have you felt anxious or worried in the past week?',
        'options': ['Rarely or never', 'A few times', 'Several times', 'Most of the time', 'Almost constantly']
    },
    {
        'id': 3,
        'question': 'How well have you been sleeping recently?',
        'options': ['Very well', 'Fairly well', 'Okay', 'Poorly', 'Very poorly']
    },
    {
        'id': 4,
        'question': 'How would you describe your energy levels?',
        'options': ['Very high', 'Moderately high', 'Average', 'Low', 'Very low']
    },
    {
        'id': 5,
        'question': 'How connected do you feel to others (friends, family, community)?',
        'options': ['Very connected', 'Moderately connected', 'Somewhat connected', 'Minimally connected', 'Very isolated']
    }
]

# Improvement suggestions based on assessment
IMPROVEMENT_SUGGESTIONS = [
    "Practice daily mindfulness or meditation for 10-15 minutes to help manage stress and improve emotional regulation.",
    "Establish a consistent sleep routine by going to bed and waking up at the same time each day, aiming for 7-9 hours of quality sleep.",
    "Engage in regular physical activity, even if it's just a 20-minute walk, as exercise releases endorphins that boost mood.",
    "Stay connected with supportive friends and family members, or consider joining a support group or community activity.",
    "Consider speaking with a mental health professional who can provide personalized guidance and support tailored to your needs."
]

def get_assessment_questions():
    """Get all assessment questions"""
    return ASSESSMENT_QUESTIONS

def get_question_by_id(question_id):
    """Get a specific question by ID"""
    for q in ASSESSMENT_QUESTIONS:
        if q['id'] == question_id:
            return q
    return None

def calculate_assessment_score(answers):
    """Calculate assessment score from answers (1-5 scale, lower is better)"""
    if not answers or len(answers) != 5:
        return None
    
    # Convert answers to scores (1 = best, 5 = worst)
    total_score = 0
    for answer in answers:
        if isinstance(answer, str):
            # Find the index of the answer in options
            for q in ASSESSMENT_QUESTIONS:
                if answer in q['options']:
                    score = q['options'].index(answer) + 1
                    total_score += score
                    break
        else:
            total_score += answer
    
    return total_score

def get_assessment_feedback(score):
    """Get feedback based on assessment score"""
    if score is None:
        return "Thank you for completing the assessment."
    
    if score <= 10:
        return "Your responses suggest you're doing well overall. Continue maintaining healthy habits and self-care practices."
    elif score <= 15:
        return "Your responses indicate some areas that could benefit from attention. Consider focusing on self-care and stress management."
    elif score <= 20:
        return "Your responses suggest you may be experiencing some challenges. It's important to prioritize your mental health and consider seeking support."
    else:
        return "Your responses indicate you may be going through a difficult time. Please consider reaching out to a mental health professional for support. Remember, seeking help is a sign of strength."

def get_improvement_suggestions():
    """Get 5 ways to improve mental health"""
    return IMPROVEMENT_SUGGESTIONS

