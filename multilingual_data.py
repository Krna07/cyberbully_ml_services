"""
Multi-language toxic words dataset for Hindi and Telugu
"""

# Hindi toxic/abusive words
HINDI_TOXIC_WORDS = [
    # Common abusive words
    'behenchod', 'madarchod', 'chutiya', 'gandu', 'harami', 'kamina',
    'kutta', 'kutti', 'saala', 'saali', 'randi', 'bhosdi', 'gaandu',
    'bhosdike', 'laude', 'lodu', 'chodu', 'bhenchod', 'mc', 'bc',
    # Insults
    'bewakoof', 'pagal', 'gadha', 'ullu', 'nalayak', 'nikamma',
    # Threats
    'maar', 'marunga', 'todenge', 'peetenge',
]

# Telugu toxic/abusive words
TELUGU_TOXIC_WORDS = [
    # Common abusive words
    'dengey', 'puka', 'gudda', 'lanjakodaka', 'lanjakoduku', 'boothu',
    'dengu', 'denguthanu', 'koduku', 'pilla', 'kukka', 'erri',
    'errihook', 'errodu', 'erripuka', 'nakodaka', 'nakoduku',
    # Insults
    'buddi', 'buddodu', 'moodu', 'moododu', 'pagal', 'mental',
    # Threats
    'kottesta', 'kottanu', 'champesta',
]

# Training data for Hindi
HINDI_TRAINING_DATA = [
    # Toxic examples
    ("tu behenchod hai", 1),
    ("madarchod kya kar raha hai", 1),
    ("chutiya insaan hai tu", 1),
    ("gandu sala", 1),
    ("harami kahi ka", 1),
    ("kutta hai tu", 1),
    ("bewakoof insaan", 1),
    ("pagal hai kya", 1),
    ("maar denge tujhe", 1),
    ("tujhe peetenge", 1),
    
    # Safe examples
    ("namaste kaise ho", 0),
    ("aap kaise hain", 0),
    ("dhanyavaad", 0),
    ("bahut achha hai", 0),
    ("mujhe pasand hai", 0),
    ("aap bahut achhe ho", 0),
    ("shukriya", 0),
    ("khush raho", 0),
    ("mil kar khushi hui", 0),
    ("aapka din shubh ho", 0),
]

# Training data for Telugu
TELUGU_TRAINING_DATA = [
    # Toxic examples
    ("nuvvu dengey", 1),
    ("lanjakodaka", 1),
    ("erripuka nuvvu", 1),
    ("kukka laga unnav", 1),
    ("nakodaka", 1),
    ("denguthanu ninnu", 1),
    ("errihook", 1),
    ("buddodu nuvvu", 1),
    ("kottesta ninnu", 1),
    ("champesta", 1),
    
    # Safe examples
    ("namaskaram ela unnaru", 0),
    ("meeru ela unnaru", 0),
    ("dhanyavadalu", 0),
    ("chala bagundi", 0),
    ("naaku istam", 0),
    ("meeru chala manchivaru", 0),
    ("santosham", 0),
    ("sukhamga undandi", 0),
    ("kalisi santosham", 0),
    ("mee roju subhamastu", 0),
]

def get_language_data(language='english'):
    """Get toxic words and training data for specified language"""
    if language == 'hindi':
        return HINDI_TOXIC_WORDS, HINDI_TRAINING_DATA
    elif language == 'telugu':
        return TELUGU_TOXIC_WORDS, TELUGU_TRAINING_DATA
    else:
        return [], []
