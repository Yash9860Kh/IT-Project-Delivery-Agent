import re
from collections import Counter

STOPWORDS = {
    'a','an','the','is','are','am','was','were','be','been','being','to','of','in','on','for','with','and','or','by','at','from','as','it','this','that','these','those','can','could','should','would','will','may','might','do','does','did','have','has','had','about','into','through','during','before','after','above','below','up','down','out','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','you','your','yours','me','my','we','our','ours','they','their','theirs','he','she','his','her','him','them','i'
}

CATEGORY_PATTERNS = {
    'general': ['project', 'software', 'system', 'process', 'delivery'],
    'planning': ['plan', 'planning', 'scope', 'timeline', 'schedule', 'milestone', 'deliverable', 'resource', 'budget', 'roadmap'],
    'agile': ['agile', 'scrum', 'sprint', 'backlog', 'standup', 'retrospective', 'iteration', 'kanban'],
    'testing': ['test', 'testing', 'qa', 'quality assurance', 'bug', 'defect', 'validation', 'verification', 'uat', 'unit test', 'integration test'],
    'documentation': ['document', 'documentation', 'report', 'srs', 'specification', 'requirement', 'design document', 'manual'],
    'coding': ['code', 'coding', 'implement', 'implementation', 'python', 'java', 'function', 'algorithm', 'debug', 'program', 'script'],
    'risk': ['risk', 'issue', 'mitigation', 'impact', 'probability', 'contingency', 'threat'],
    'compliance': ['compliance', 'privacy', 'security', 'dpdp', 'it act', 'legal', 'policy', 'data protection', 'regulation']
}

FOLLOWUP_HINTS = {
    'it', 'that', 'this', 'those', 'these', 'they', 'them',
    'same', 'above', 'previous', 'earlier', 'continue', 'also',
    'more', 'explain'
}

def clean_text(text):
    text = (text or '').lower().strip()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def tokenize(text):
    return [token for token in clean_text(text).split() if token and token not in STOPWORDS]

def extract_keywords(text, top_n=6):
    tokens = tokenize(text)
    if not tokens:
        return []
    return [word for word, _ in Counter(tokens).most_common(top_n)]

def score_categories(text):
    cleaned = clean_text(text)
    scores = {category: 0 for category in CATEGORY_PATTERNS}

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern in cleaned:
                scores[category] += 2 if ' ' in pattern else 1

    return scores

def detect_followup(text):
    tokens = set(tokenize(text))
    if len(tokens) <= 4:
        return True
    return bool(tokens.intersection(FOLLOWUP_HINTS))

def detect_category(text, previous_context=''):
    current_scores = score_categories(text)
    context_scores = score_categories(previous_context) if previous_context else {
        k: 0 for k in CATEGORY_PATTERNS
    }

    merged_scores = {}
    for key in CATEGORY_PATTERNS:
        merged_scores[key] = current_scores.get(key, 0) + (0.5 * context_scores.get(key, 0))

    best_category = max(merged_scores, key=merged_scores.get)

    # Tie-breaking: if 'general' ties with a specific category, prefer the specific one
    if best_category == 'general':
        specific = {k: v for k, v in merged_scores.items() if k != 'general'}
        alt = max(specific, key=specific.get)
        if specific[alt] >= merged_scores['general']:
            best_category = alt

    best_score = merged_scores[best_category]

    if best_score == 0:
        return 'general', 0, merged_scores

    return best_category, best_score, merged_scores

def build_context_summary(history_messages, max_items=4):
    if not history_messages:
        return ''

    recent = history_messages[-max_items:]
    combined = ' '.join(msg.get('content') or '' for msg in recent)
    return clean_text(combined)