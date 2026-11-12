from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import os
import sys

# Add src/ directory to python path to resolve local imports correctly
sys.path.append(str(Path(__file__).resolve().parent))

from nlp_utils import detect_category, build_context_summary, detect_followup, extract_keywords

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_FILE = BASE_DIR / "prompts" / "prompt_library.md"
STANDARDS_FILE = BASE_DIR / "data" / "standards" / "it_standards.md"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)

MODEL_NAME = "gemini-3.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

def load_file_content(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    return file_path.read_text(encoding="utf-8")

def get_prompt_section(prompt_text, heading):
    sections = prompt_text.split("## ")
    for section in sections:
        if section.strip().startswith(heading):
            return "## " + section.strip()
    return None

def map_category_to_heading(category):
    mapping = {
        "general": "1. Base Project Guidance Prompt",
        "planning": "2. Project Planning Prompt",
        "agile": "3. Agile / Scrum Prompt",
        "testing": "4. Testing and QA Prompt",
        "documentation": "5. Documentation Prompt",
        "coding": "6. Coding Standards Prompt",
        "risk": "7. Risk Management Prompt",
        "compliance": "8. Compliance and Data Protection Prompt"
    }
    return mapping.get(category, "1. Base Project Guidance Prompt")

def fill_prompt_template(template, user_query):
    placeholders = {
        "{phase}": "Requirement Analysis",
        "{methodology}": "Agile",
        "{user_role}": "Project Team Member",
        "{module_name}": "General Module",
        "{project_type}": "Web Application",
        "{team_size}": "5",
        "{duration}": "8 Weeks",
        "{sprint_number}": "Sprint 1",
        "{activity_type}": "Sprint Planning",
        "{testing_type}": "Functional Testing",
        "{document_type}": "SRS",
        "{audience}": "Project Team",
        "{language}": "Python",
        "{risk_category}": "Technical Risk",
        "{impact_level}": "Medium",
        "{data_type}": "Personal Data",
        "{project_area}": "Project Documentation",
        "{compliance_type}": "Privacy and Security",
        "{user_query}": user_query
    }

    for key, value in placeholders.items():
        template = template.replace(key, value)

    return template

def build_final_prompt(user_query, history_messages=None):
    prompt_library = load_file_content(PROMPT_FILE)
    standards_text = load_file_content(STANDARDS_FILE)

    previous_context = build_context_summary(history_messages or [])
    category, score, merged_scores = detect_category(user_query, previous_context)

    if score == 0 and detect_followup(user_query) and previous_context:
        category = "general"

    heading = map_category_to_heading(category)
    template = get_prompt_section(prompt_library, heading)

    if not template:
        raise ValueError(f"Prompt section not found for category: {category}")

    filled_prompt = fill_prompt_template(template, user_query)

    final_prompt = f"""
You are an AI-Powered IT Project Delivery Agent.

Your job is to answer project-related queries in a clear, structured, concise, and professional way.

Use the following IT standards and knowledge base as reference:

{standards_text}

Use the following task-specific prompt template:

{filled_prompt}

Instructions:
- Give a professional and accurate answer.
- Keep the answer relevant to the user query.
- Use proper project delivery language.
- If needed, give steps, bullet points, or structured sections.
- Do not give unrelated information.
"""
    return final_prompt, category, merged_scores

def get_model_response(user_query, history_messages=None):
    final_prompt, category, merged_scores = build_final_prompt(user_query, history_messages)
    response = model.generate_content(final_prompt)

    if hasattr(response, "text") and response.text:
        return response.text, category, final_prompt, merged_scores

    return "No response generated.", category, final_prompt, merged_scores

def list_available_models():
    print("\nModels that support generateContent:\n")
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", [])
        if "generateContent" in methods:
            print(m.name)

def main():
    print("AI-Powered IT Project Delivery Agent")
    print(f"Using model: {MODEL_NAME}")
    print("Enter a project-related query.\n")

    user_query = input("Query: ").strip()

    if user_query.lower() == "list models":
        list_available_models()
        return

    if not user_query:
        print("No query entered.")
        return

    try:
        answer, category, final_prompt, merged_scores = get_model_response(user_query)
        print("\nResponse:\n")
        print(answer)
        print(f"\nDetected category: {category}")
    except Exception as e:
        print("\nError occurred:")
        print(type(e).__name__, "-", str(e))
        print("\nTip: type 'list models' to see supported model names.")

if __name__ == "__main__":
    main()