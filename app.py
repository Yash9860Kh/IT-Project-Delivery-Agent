from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import streamlit.components.v1 as components
import os
import uuid
import json
from io import BytesIO
from datetime import datetime

from src.nlp_utils import detect_category, extract_keywords, detect_followup, build_context_summary

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROMPT_FILE = BASE_DIR / "prompts" / "prompt_library.md"
STANDARDS_FILE = BASE_DIR / "data" / "standards" / "it_standards.md"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)

MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

TOPIC_QUICK_LINKS = [("Planning", "planning"), ("Agile", "agile"), ("Testing", "testing"), ("Documentation", "documentation"), ("Coding", "coding"), ("Risk", "risk"), ("Compliance", "compliance")]
DOMAINS = ["Planning", "Agile", "Testing", "Documentation", "Coding", "Risk", "Compliance", "General"]
STATIC_INTERN_USERS = {
}

TOPIC_ROLE_MAP = {
    "Planning": ["Project Manager", "Business Analyst", "Team Lead"],
    "Agile": ["Scrum Master", "Product Owner", "Developer"],
    "Testing": ["QA Engineer", "Test Lead", "Automation Engineer"],
    "Documentation": ["Business Analyst", "Technical Writer", "Project Coordinator"],
    "Coding": ["Backend Developer", "Frontend Developer", "Full Stack Developer"],
    "Risk": ["Project Manager", "Risk Analyst", "Delivery Manager"],
    "Compliance": ["Compliance Officer", "Security Analyst", "Project Manager"],
}
DOMAIN_TOPIC_MAP = {
    "Planning": ["Scope Definition", "Scheduling", "Resource Allocation", "Milestone Tracking", "Budgeting"],
    "Agile": ["Sprint Planning", "Backlog Refinement", "Daily Standup", "Sprint Review", "Sprint Retrospective"],
    "Testing": ["Unit Testing", "Integration Testing", "System Testing", "Regression Testing", "UAT", "Performance Testing", "Security Testing"],
    "Documentation": ["SRS Document", "System Design Document", "Test Plan", "User Manual", "Release Notes", "Meeting Minutes", "Status Report"],
    "Coding": ["Clean Code", "Formatting & Naming", "Modularity", "Error Handling", "Code Review Checklist"],
    "Risk": ["Risk Identification", "Impact Assessment", "Mitigation Strategy", "Contingency Planning", "Issue Escalation"],
    "Compliance": ["DPDP Act 2023", "IT Act 2000", "Data Minimization", "Security Safeguards", "Access Control"],
    "General": ["SDLC Phases", "Project Delivery Overview"]
}

DOMAIN_ROLE_MAP = {
    "Planning": ["Project Manager", "Business Analyst", "Team Lead"],
    "Agile": ["Scrum Master", "Product Owner", "Developer"],
    "Testing": ["QA Engineer", "Test Lead", "Automation Engineer"],
    "Documentation": ["Business Analyst", "Technical Writer", "Project Coordinator"],
    "Coding": ["Backend Developer", "Frontend Developer", "Full Stack Developer"],
    "Risk": ["Project Manager", "Risk Analyst", "Delivery Manager"],
    "Compliance": ["Compliance Officer", "Security Analyst", "Project Manager"],
    "General": ["Project Team Member", "Stakeholder"]
}
PLACEHOLDERS = {
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
    "{compliance_type}": "Privacy and Security"
}


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


def fill_prompt_template(template, user_query, context_summary="", followup=False, keywords=None):
    placeholders = dict(PLACEHOLDERS)
    placeholders["{user_query}"] = user_query

    for key, value in placeholders.items():
        template = template.replace(key, value)

    context_note = context_summary if context_summary else "No prior conversation context."
    keyword_note = ", ".join(keywords) if keywords else "None"
    followup_note = "Yes" if followup else "No"

    template += (
        f"\n\nAdditional Context:\n"
        f"- Previous conversation summary: {context_note}\n"
        f"- Detected keywords: {keyword_note}\n"
        f"- Follow-up query: {followup_note}\n"
    )
    return template


def build_final_prompt(user_query, history_messages=None):
    history_messages = history_messages or []

    prompt_library = load_file_content(PROMPT_FILE)
    standards_text = load_file_content(STANDARDS_FILE)

    context_summary = build_context_summary(history_messages)
    category, category_score, category_scores = detect_category(user_query, context_summary)
    heading = map_category_to_heading(category)
    template = get_prompt_section(prompt_library, heading)

    if not template:
        raise ValueError(f"Prompt section not found for category: {category}")

    keywords = extract_keywords(user_query)
    followup = detect_followup(user_query)
    filled_prompt = fill_prompt_template(
        template,
        user_query,
        context_summary=context_summary,
        followup=followup,
        keywords=keywords
    )

    final_prompt = f"""
You are an AI-Powered IT Project Delivery Agent.

Your role is to answer IT project delivery questions in a concise, professional, structured, and accurate way.

Use the following IT standards and knowledge base:

{standards_text}

Use the following prompt template:

{filled_prompt}

Instructions:
- Keep the answer relevant to the user query.
- Use proper IT project delivery language.
- Present the answer clearly.
- Use bullets or sections where useful.
- Avoid unnecessary extra information.
- Use the previous conversation only when it improves relevance.
"""

    debug_info = {
        "category": category,
        "category_score": category_score,
        "category_scores": category_scores,
        "keywords": keywords,
        "followup_detected": followup,
        "context_summary": context_summary,
        "prompt_heading": heading
    }

    return final_prompt, category, debug_info


def get_model_response(user_query, history_messages=None):
    final_prompt, category, debug_info = build_final_prompt(user_query, history_messages=history_messages)
    response = model.generate_content(final_prompt)

    if hasattr(response, "text") and response.text:
        return response.text, category, final_prompt, debug_info

    return "No response generated.", category, final_prompt, debug_info


def transcribe_audio_bytes(audio_bytes, mime_type="audio/wav"):
    # gemini-1.5-flash-latest supports audio via generateContent on v1
    transcribe_model = genai.GenerativeModel("gemini-3.5-flash-latest")
    response = transcribe_model.generate_content([
        "Transcribe the following audio accurately into English text. Return ONLY the transcription text, without any quotes, introductory comments, or extra explanations.",
        {"mime_type": mime_type, "data": audio_bytes}
    ])
    if hasattr(response, "text") and response.text:
        return response.text.strip()
    return ""


def create_chat_name(first_message=None):
    if first_message:
        text = " ".join(first_message.strip().split())
        return text[:38].rstrip() + ("..." if len(text) > 38 else "")
    return f"New Chat {datetime.now().strftime('%H:%M')}"


def validate_static_login(intern_id, password):
    user = STATIC_INTERN_USERS.get((intern_id or "").strip().upper())
    if user and user["password"] == password:
        return {"intern_id": (intern_id or "").strip().upper(), **user}
    return None


# Future PostgreSQL sample for login validation. Replace placeholders later.
# import psycopg2
# def validate_login_from_postgres(intern_id, password):
#     conn = psycopg2.connect(
#         host="YOUR_HOST",
#         port="5432",
#         dbname="YOUR_DB_NAME",
#         user="YOUR_DB_USER",
#         password="YOUR_DB_PASSWORD"
#     )
#     cur = conn.cursor()
#     cur.execute(
#         "SELECT intern_id, full_name, role FROM intern_users WHERE intern_id = %s AND password = %s",
#         (intern_id, password)
#     )
#     row = cur.fetchone()
#     cur.close()
#     conn.close()
#     if row:
#         return {"intern_id": row[0], "name": row[1], "role": row[2]}
#     return None


def render_login_gate():
    # render_base_css already calls get_theme_css() — call only once
    render_base_css()

    # ── Extra login-page scoped styles ────────────────────────────────────
    st.markdown(
        """
        <style>
        /* Centre the login card on the page */
        .login-outer {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding-top: 2.5rem;
        }
        .login-card {
            width: 100%;
            max-width: 460px;
            background: linear-gradient(160deg, rgba(18,26,50,0.92) 0%, rgba(10,14,28,0.90) 100%);
            border: 1px solid rgba(80,110,220,0.18);
            border-radius: 24px;
            padding: 2rem 2rem 1.6rem 2rem;
            box-shadow: 0 20px 50px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.04) inset;
            backdrop-filter: blur(20px);
        }
        .login-logo {
            font-size: 2rem;
            text-align: center;
            margin-bottom: 0.35rem;
        }
        .login-title {
            font-size: 1.18rem;
            font-weight: 700;
            color: #edf0ff;
            text-align: center;
            letter-spacing: -0.02em;
            margin-bottom: 0.3rem;
        }
        .login-subtitle {
            font-size: 0.72rem;
            color: #7a96c8;
            text-align: center;
            line-height: 1.5;
            margin-bottom: 1.4rem;
        }
        .login-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(80,110,220,0.22), transparent);
            margin: 1rem 0;
        }
        .login-hint {
            font-size: 0.68rem;
            color: #5a78cc;
            text-align: center;
            margin-top: 0.8rem;
            line-height: 1.6;
        }
        /* Constrain Streamlit tab bar inside the card */
        [data-testid="stTabs"] {
            margin-top: 0 !important;
        }
        /* Make inputs and buttons look right inside the narrow card */
        .login-card .stTextInput input {
            border-radius: 12px !important;
            font-size: 0.82rem !important;
            padding: 0.55rem 0.9rem !important;
        }
        .login-card .stButton > button {
            border-radius: 13px !important;
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            padding: 0.6rem 0 !important;
            background: linear-gradient(135deg, rgba(60,80,180,0.70), rgba(80,40,160,0.65)) !important;
            border: 1px solid rgba(100,130,255,0.30) !important;
            color: #d0e0ff !important;
        }
        .login-card .stButton > button:hover {
            border-color: rgba(120,160,255,0.50) !important;
            box-shadow: 0 0 18px rgba(80,120,255,0.22) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Centred wrapper ────────────────────────────────────────────────────
    _, centre_col, _ = st.columns([1, 2, 1])

    with centre_col:
        st.markdown(
            """
            <div class="login-logo">🤖</div>
            <div class="login-title">IT Project Delivery Agent</div>
            <div class="login-subtitle">
                Sign in with your intern credentials to access the workspace.<br>
                Create an account in the Register tab to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["🔑  Login", "📝  Register"])

        with tab_login:
            intern_id = st.text_input(
                "Intern ID", placeholder="e.g. INT001", key="login_id"
            )
            password = st.text_input(
                "Password", type="password", placeholder="Enter password", key="login_password"
            )

            st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

            if st.button("Login", use_container_width=True, key="login_btn"):
                clean_id = (intern_id or "").strip().upper()
                user = st.session_state.custom_users.get(clean_id)
                if not user:
                    user = validate_static_login(intern_id, password)
                else:
                    if user["password"] != password:
                        user = None

                if user:
                    st.session_state.is_authenticated = True
                    st.session_state.logged_in_user = user
                    uid = user["intern_id"]

                    if uid not in st.session_state.user_chats:
                        first_chat_id = str(uuid.uuid4())
                        st.session_state.user_chats[uid] = {
                            first_chat_id: {
                                "title": "New Chat",
                                "messages": [],
                                "created_at": datetime.now().strftime("%d %b, %H:%M"),
                            }
                        }

                    st.session_state.chat_sessions = st.session_state.user_chats[uid]
                    st.session_state.current_chat_id = next(
                        iter(st.session_state.chat_sessions.keys())
                    )
                    st.session_state.login_activity = [
                        {
                            "time": datetime.now().strftime("%d %b, %H:%M:%S"),
                            "event": "Login successful",
                            "chat": f"{user['intern_id']} · {user['name']}",
                        }
                    ]
                    st.rerun()
                else:
                    st.error("Invalid intern ID or password.")

            st.markdown(
                """
                <div class="login-hint">
                    No preset demo accounts are configured.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab_register:
            reg_id = st.text_input(
                "Choose Intern ID", placeholder="e.g. INT004", key="reg_id"
            )
            reg_name = st.text_input(
                "Full Name", placeholder="e.g. Aarav Sharma", key="reg_name"
            )
            reg_role = st.selectbox(
                "Role",
                ["QA Intern", "Business Analyst Intern", "Project Intern", "Developer Intern"],
                key="reg_role",
            )
            reg_password = st.text_input(
                "Choose Password",
                type="password",
                placeholder="Enter a secure password",
                key="reg_password",
            )

            st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

            if st.button("Create Account", use_container_width=True, key="register_btn"):
                clean_id = (reg_id or "").strip().upper()
                if not clean_id or not reg_name or not reg_password:
                    st.error("Please fill in all fields.")
                elif clean_id in STATIC_INTERN_USERS or clean_id in st.session_state.custom_users:
                    st.error("Intern ID already exists. Choose a different one.")
                else:
                    st.session_state.custom_users[clean_id] = {
                        "intern_id": clean_id,
                        "name": reg_name.strip(),
                        "role": reg_role,
                        "password": reg_password,
                    }
                    st.success("Account created! Switch to the Login tab to sign in.")


def ensure_chat_state():
    if "custom_users" not in st.session_state:
        st.session_state.custom_users = {}
    if "user_chats" not in st.session_state:
        st.session_state.user_chats = {}

    if "chat_sessions" not in st.session_state:
        first_chat_id = str(uuid.uuid4())
        st.session_state.chat_sessions = {
            first_chat_id: {
                "title": "New Chat",
                "messages": [],
                "created_at": datetime.now().strftime("%d %b, %H:%M")
            }
        }
        st.session_state.current_chat_id = first_chat_id

    if "login_activity" not in st.session_state:
        st.session_state.login_activity = []
        st.session_state.login_activity.append({
            "time": datetime.now().strftime("%d %b, %H:%M:%S"),
            "event": "Session started",
            "chat": st.session_state.chat_sessions[st.session_state.current_chat_id]["title"]
        })

    defaults = {
        "last_prompt": "",
        "last_response": "",
        "last_category": "",
        "last_debug": {},
        "search_chats": "",
        "selected_topic": None,
        "selected_domain": None,
        "selected_role": None,
        "context_decisions": {},
        "show_prompt_toggle": False,
        "show_debug_toggle": False,
        "pending_prompt": None,
        "copied_message_id": "",
        "is_authenticated": False,
        "logged_in_user": None,
        "last_processed_audio": None,
        "voice_transcription": ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_current_chat():
    return st.session_state.chat_sessions[st.session_state.current_chat_id]


def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[chat_id] = {
        "title": f"New Chat {len(st.session_state.chat_sessions) + 1}",
        "messages": [],
        "created_at": datetime.now().strftime("%d %b, %H:%M")
    }
    st.session_state.current_chat_id = chat_id
    st.session_state.login_activity.append({
        "time": datetime.now().strftime("%d %b, %H:%M:%S"),
        "event": "New chat created",
        "chat": st.session_state.chat_sessions[chat_id]["title"]
    })
    # FUTURE POSTGRESQL CODE FOR NEW CHAT CREATION:
    # """
    # try:
    #     conn = psycopg2.connect(host="YOUR_HOST", port="5432", dbname="YOUR_DB_NAME", user="YOUR_DB_USER", password="YOUR_DB_PASSWORD")
    #     cur = conn.cursor()
    #     cur.execute(
    #         "INSERT INTO chats (chat_id, intern_id, title, created_at) VALUES (%s, %s, %s, %s)",
    #         (chat_id, st.session_state.logged_in_user["intern_id"], f"New Chat {len(st.session_state.chat_sessions)}", datetime.now())
    #     )
    #     conn.commit()
    #     cur.close()
    #     conn.close()
    # except Exception as e:
    #     st.error(f"DB Error: {e}")
    # """


def rename_current_chat(new_title):
    current_id = st.session_state.current_chat_id
    cleaned = " ".join((new_title or "").split())
    if cleaned:
        st.session_state.chat_sessions[current_id]["title"] = cleaned[:60]
        st.session_state.login_activity.append({
            "time": datetime.now().strftime("%d %b, %H:%M:%S"),
            "event": "Chat renamed",
            "chat": st.session_state.chat_sessions[current_id]["title"]
        })
        # FUTURE POSTGRESQL CODE FOR CHAT RENAMING:
        # """
        # try:
        #     conn = psycopg2.connect(...)
        #     cur = conn.cursor()
        #     cur.execute("UPDATE chats SET title = %s WHERE chat_id = %s", (cleaned[:60], current_id))
        #     conn.commit()
        #     cur.close()
        #     conn.close()
        # except Exception as e:
        #     st.error(f"DB Error: {e}")
        # """


def export_chat_text(chat):
    lines = [
        f"Chat title: {chat['title']}",
        f"Created: {chat['created_at']}",
        ""
    ]
    for message in chat.get("messages", []):
        role = message.get("role", "assistant").upper()
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
        lines.append("")
    return "\n".join(lines)


def delete_current_chat():
    current_id = st.session_state.current_chat_id
    if len(st.session_state.chat_sessions) == 1:
        st.session_state.chat_sessions[current_id]["messages"] = []
        st.session_state.chat_sessions[current_id]["title"] = "New Chat"
        st.session_state.login_activity.append({
            "time": datetime.now().strftime("%d %b, %H:%M:%S"),
            "event": "Chat cleared",
            "chat": "New Chat"
        })
        # FUTURE POSTGRESQL CODE FOR CHAT CLEARING:
        # """
        # try:
        #     conn = psycopg2.connect(...)
        #     cur = conn.cursor()
        #     cur.execute("DELETE FROM messages WHERE chat_id = %s", (current_id,))
        #     cur.execute("UPDATE chats SET title = 'New Chat' WHERE chat_id = %s", (current_id,))
        #     conn.commit()
        #     cur.close()
        #     conn.close()
        # except Exception as e:
        #     st.error(f"DB Error: {e}")
        # """
        return
    removed_title = st.session_state.chat_sessions[current_id]["title"]
    del st.session_state.chat_sessions[current_id]
    st.session_state.login_activity.append({
        "time": datetime.now().strftime("%d %b, %H:%M:%S"),
        "event": "Chat deleted",
        "chat": removed_title
    })
    st.session_state.current_chat_id = next(iter(st.session_state.chat_sessions.keys()))
    # FUTURE POSTGRESQL CODE FOR CHAT DELETION:
    # """
    # try:
    #     conn = psycopg2.connect(...)
    #     cur = conn.cursor()
    #     cur.execute("DELETE FROM messages WHERE chat_id = %s", (current_id,))
    #     cur.execute("DELETE FROM chats WHERE chat_id = %s", (current_id,))
    #     conn.commit()
    #     cur.close()
    #     conn.close()
    # except Exception as e:
    #     st.error(f"DB Error: {e}")
    # """


def save_feedback(chat_id, message_id):
    widget_key = f"feedback_{chat_id}_{message_id}"
    selected = st.session_state.get(widget_key)
    messages = st.session_state.chat_sessions[chat_id]["messages"]
    for msg in messages:
        if msg.get("id") == message_id:
            msg["feedback"] = selected
            break
    # FUTURE POSTGRESQL CODE FOR SAVING FEEDBACK:
    # """
    # try:
    #     conn = psycopg2.connect(...)
    #     cur = conn.cursor()
    #     cur.execute("UPDATE messages SET feedback = %s WHERE message_id = %s AND chat_id = %s", (selected, message_id, chat_id))
    #     conn.commit()
    #     cur.close()
    #     conn.close()
    # except Exception as e:
    #     st.error(f"DB Error: {e}")
    # """


def save_context_choice(chat_id, message_id):
    widget_key = f"context_choice_{chat_id}_{message_id}"
    selected = st.session_state.get(widget_key)
    messages = st.session_state.chat_sessions[chat_id]["messages"]
    for msg in messages:
        if msg.get("id") == message_id:
            msg["store_context"] = selected
            break


def set_quick_prompt(prompt_text):
    st.session_state.pending_prompt = prompt_text


def render_copy_button(text, key):
    button_id = f"copy_btn_{key}"
    safe_text = json.dumps(text)
    components.html(
        f"""
        <div style='display:flex;justify-content:flex-end;margin-top:6px;'>
            <button id='{button_id}'
                style='
                    background: linear-gradient(135deg, rgba(38,50,90,0.80), rgba(22,28,56,0.85));
                    color: #a8c0ff;
                    border: 1px solid rgba(80,110,220,0.22);
                    border-radius: 10px;
                    padding: 6px 12px;
                    font-size: 11px;
                    font-weight: 600;
                    cursor: pointer;
                    letter-spacing: 0.03em;'>
                Copy answer
            </button>
        </div>
        <script>
        document.getElementById("{button_id}").addEventListener("click", function() {{
            navigator.clipboard.writeText({safe_text}).then(() => {{
                this.innerText = "Copied ✓";
            }});
        }});
        </script>
        """,
        height=42,
    )


def get_theme_css():
    return """
    <style>
    /* Force Dark Theme Variables at the Root Level */
    :root {
        --background-color: #07090f !important;
        --secondary-background-color: #0a0e18 !important;
        --text-color: #dde4f4 !important;
        --primary-color: #5a8fff !important;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #07090f !important;
        background: 
            radial-gradient(ellipse 80% 60% at 10% 0%, rgba(59,80,180,0.13) 0%, transparent 60%),
            radial-gradient(ellipse 60% 50% at 90% 100%, rgba(120,60,200,0.10) 0%, transparent 55%),
            linear-gradient(160deg, #07090f 0%, #0e1420 45%, #080c14 100%) !important;
        color: #dde4f4 !important;
    }

    /* ── Sidebar forced dark theme ── */
    [data-testid="stSidebar"], [data-testid="stSidebarUserContent"] {
        background-color: #070a10 !important;
        background: linear-gradient(180deg, #070a10 0%, #0a0e18 100%) !important;
        border-right: 1px solid rgba(90,110,220,0.16) !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }

    /* ── Glass cards ── */
    .hero-card {
        background: linear-gradient(145deg, rgba(22,30,52,0.88) 0%, rgba(12,16,28,0.82) 100%) !important;
        border: 1px solid rgba(90,130,255,0.14) !important;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.04) inset, 0 16px 40px rgba(0,0,0,0.45) !important;
        backdrop-filter: blur(18px);
        color: #e8eaf0 !important;
    }
    .glass-panel, .sidebar-card {
        background: linear-gradient(160deg, rgba(18,24,42,0.82) 0%, rgba(10,14,24,0.78) 100%) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35) !important;
        backdrop-filter: blur(14px);
        color: #e8eaf0 !important;
    }
    .topic-card, .query-card {
        background: linear-gradient(160deg, rgba(16,22,40,0.80) 0%, rgba(10,14,24,0.76) 100%) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        box-shadow: 0 4px 18px rgba(0,0,0,0.30) !important;
        backdrop-filter: blur(12px);
        color: #e8eaf0 !important;
    }

    /* ── Typography ── */
    .hero-title    { color: #edf0ff !important; text-shadow: 0 1px 12px rgba(100,140,255,0.18) !important; }
    .metric-value  { color: #c8d8ff !important; }
    .hero-subtitle { color: #8ca0c8 !important; }
    .small-note    { color: #8ca0c8 !important; }
    .section-label { color: #5a78cc !important; letter-spacing: 0.10em; }
    .chat-toolbar-title { color: #6a88d8 !important; }
    .metric-label  { color: #5a78cc !important; }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.018) !important;
        border: 1px solid rgba(255,255,255,0.055) !important;
        border-radius: 18px !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(60,90,200,0.07) !important;
        border-color: rgba(80,110,230,0.12) !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span,
    .stMarkdown, .stCaption, .stCheckbox label,
    .stSelectbox label, .stTextInput label {
        color: #dde4f4 !important;
    }

    /* ── Buttons ── */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(160deg, rgba(38,48,82,0.82) 0%, rgba(22,28,50,0.88) 100%) !important;
        color: #c8d8ff !important;
        border: 1px solid rgba(80,110,220,0.22) !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(100,140,255,0.42) !important;
        box-shadow: 0 0 14px rgba(80,120,255,0.18) !important;
    }

    /* ── Inputs & selects ── */
    .stTextInput input,
    [data-testid="stChatInput"] textarea,
    [data-baseweb="select"] > div {
        background: rgba(14,18,32,0.96) !important;
        color: #dde4f4 !important;
        border: 1px solid rgba(80,110,220,0.20) !important;
        border-radius: 12px !important;
    }
    .stTextInput input:focus,
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(100,140,255,0.45) !important;
        box-shadow: 0 0 0 3px rgba(80,120,255,0.10) !important;
    }
    [data-baseweb="select"] input,
    [data-baseweb="select"] [data-testid="stSelectboxValue"] { color: #dde4f4 !important; }

    /* ── Sidebar collapse button ── */
    [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] {
        background: rgba(18,24,42,0.92) !important;
        color: #8ca0c8 !important;
        border: 1px solid rgba(80,110,220,0.18) !important;
        border-radius: 12px !important;
    }

    /* ── Info box ── */
    .stInfo {
        background: rgba(40,60,130,0.18) !important;
        border: 1px solid rgba(80,120,220,0.22) !important;
        color: #a8c0f0 !important;
        border-radius: 12px !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tab"] {
        color: #6a88c8 !important;
        font-size: 0.80rem !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #a8c4ff !important;
        border-bottom: 2px solid rgba(100,150,255,0.55) !important;
    }

    /* ── Audio input ── */
    [data-testid="stAudioInput"] {
        background: rgba(14,18,32,0.82) !important;
        border: 1px solid rgba(80,110,220,0.20) !important;
        border-radius: 14px !important;
        padding: 0.4rem 0.6rem !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(80,110,200,0.28); border-radius: 6px; }
    </style>
    """


def render_base_css():
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        /* ── Layout ── */
        .block-container {
            max-width: 1140px;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 0.8rem;
            padding-bottom: 1rem;
        }

        /* ── Hero card ── */
        .hero-card {
            border-radius: 24px;
            padding: 1.25rem 1.35rem 1rem 1.35rem;
            margin-bottom: 0.9rem;
        }
        .hero-title {
            font-size: 1.28rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 0.22rem;
        }
        .hero-subtitle {
            font-size: 0.74rem;
            line-height: 1.5;
            margin-bottom: 1rem;
            max-width: 72ch;
        }

        /* ── Metric chips ── */
        .metric-row {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
        }
        .metric-chip {
            border-radius: 14px;
            padding: 0.62rem 0.76rem;
            background: linear-gradient(135deg, rgba(40,55,100,0.30) 0%, rgba(20,28,55,0.28) 100%);
            border: 1px solid rgba(80,110,220,0.16);
        }
        .metric-label {
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            font-size: 0.74rem;
            font-weight: 600;
        }

        /* ── Section labels ── */
        .section-label {
            font-size: 0.64rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin: 0.3rem 0 0.48rem 0;
        }

        /* ── Sidebar / glass cards ── */
        .sidebar-card, .glass-panel, .topic-card {
            border-radius: 18px;
            padding: 0.88rem;
            margin-bottom: 0.48rem;
        }
        .small-note {
            font-size: 0.73rem;
            line-height: 1.5;
        }

        /* ── Chat toolbar ── */
        .chat-toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.8rem;
            margin: 0.15rem 0 0.6rem 0;
            padding: 0.5rem 0.75rem;
            border-radius: 14px;
            background: rgba(255,255,255,0.018);
            border: 1px solid rgba(255,255,255,0.045);
        }
        .chat-toolbar-title {
            font-size: 0.80rem;
            font-weight: 500;
        }

        /* ── Topic grid ── */
        .topic-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }

        /* ── Chat messages ── */
        [data-testid="stChatMessage"] {
            padding: 0.52rem 0.68rem 0.48rem 0.68rem;
            margin-bottom: 0.38rem;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
            font-size: 0.83rem !important;
            line-height: 1.55 !important;
        }

        /* ── Buttons ── */
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 13px !important;
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            padding-top: 0.48rem !important;
            padding-bottom: 0.48rem !important;
        }

        /* ── Text inputs ── */
        .stTextInput input {
            font-size: 0.80rem !important;
            border-radius: 12px !important;
            padding: 0.55rem 0.85rem !important;
        }
        [data-testid="stChatInput"] textarea {
            font-size: 0.83rem !important;
        }

        /* ── Voice badge ── */
        .voice-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.68rem;
            font-weight: 600;
            padding: 0.26rem 0.6rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(60,90,200,0.22), rgba(100,60,200,0.18));
            border: 1px solid rgba(100,140,255,0.28);
            color: #a8c0ff;
            margin-bottom: 0.3rem;
        }
        .voice-transcribed {
            background: linear-gradient(135deg, rgba(20,50,120,0.18), rgba(40,20,100,0.14));
            border: 1px solid rgba(80,120,220,0.22);
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
            font-size: 0.80rem;
            color: #c0d0f4;
            margin-bottom: 0.45rem;
            font-style: italic;
        }

        /* ── Mini badge ── */
        .mini-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.66rem;
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            background: rgba(80,120,255,0.10);
            border: 1px solid rgba(80,120,255,0.18);
            color: #90b0ff;
            margin-right: 0.35rem;
        }

        /* ── Typing loader ── */
        .typing-loader {
            display: flex;
            align-items: center;
            gap: 0.42rem;
            padding: 0.4rem 0.1rem 0.7rem 0.1rem;
        }
        .typing-dot {
            width: 7px; height: 7px;
            border-radius: 999px;
            background: linear-gradient(135deg, #5a8fff, #9060ff);
            animation: blink 1.2s infinite ease-in-out;
        }
        .typing-dot:nth-child(2) { animation-delay: 0.16s; }
        .typing-dot:nth-child(3) { animation-delay: 0.32s; }
        @keyframes blink {
            0%, 80%, 100% { opacity: 0.22; transform: translateY(0) scale(0.9); }
            40% { opacity: 1; transform: translateY(-3px) scale(1.1); }
        }

        /* ── Responsive ── */
        @media (max-width: 900px) {
            .metric-row { grid-template-columns: 1fr; }
            .topic-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .hero-title { font-size: 1.18rem; }
            .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
        }
        @media (max-width: 600px) {
            .topic-grid { grid-template-columns: 1fr 1fr; gap: 0.42rem; }
            .hero-subtitle { font-size: 0.82rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(
    page_title="IT Project Delivery Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

ensure_chat_state()

if not st.session_state.is_authenticated:
    render_login_gate()
    st.stop()

render_base_css()

with st.sidebar:
    user = st.session_state.logged_in_user or {}
    if user:
        user_html = f'<div class="sidebar-card"><div class="small-note"><b>Logged in as</b><br>{user.get("name", "Unknown")}<br>{user.get("intern_id", "")} · {user.get("role", "")}</div></div>'
        st.markdown(user_html, unsafe_allow_html=True)
    st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-card"><div style="font-size:1rem;font-weight:700;">IT Project Delivery Agent</div><div class="small-note">A clean, modern workspace with topic shortcuts, reusable chat history, copy tools, and assistant feedback controls.</div></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("+  New chat", use_container_width=True):
            create_new_chat()
            st.rerun()
    with col_b:
        if st.button("🗑️  Delete", use_container_width=True):
            delete_current_chat()
            st.rerun()

    current_chat = get_current_chat()

    st.text_input("Search chats", key="search_chats", placeholder="Search by title...", label_visibility="collapsed")

    st.markdown('<div class="section-label">Quick topics</div>', unsafe_allow_html=True)
    quick_topic = st.selectbox(
        "Open a quick topic",
        options=[label for label, _ in TOPIC_QUICK_LINKS],
        index=None,
        placeholder="Choose a quick topic",
        key="sidebar_quick_topic"
    )
    if st.button("Go", use_container_width=True) and quick_topic:
        set_quick_prompt(f"Give concise professional guidance for {quick_topic.lower()} in IT project delivery.")
        st.rerun()

    st.markdown('<div class="section-label">History</div>', unsafe_allow_html=True)
    filter_text = st.session_state.search_chats.strip().lower()
    chat_items = list(st.session_state.chat_sessions.items())
    shown = 0
    for chat_id, chat_data in reversed(chat_items):
        title = chat_data["title"]
        if filter_text and filter_text not in title.lower():
            continue
        label = f"{title}\n{chat_data.get('created_at', '')}"
        if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.current_chat_id = chat_id
            st.session_state.login_activity.append({
                "time": datetime.now().strftime("%d %b, %H:%M:%S"),
                "event": "Chat opened",
                "chat": st.session_state.chat_sessions[chat_id]["title"]
            })
            st.rerun()
        shown += 1
    if shown == 0:
        st.caption("No matching chats found.")


    st.markdown('<div class="section-label">Status</div>', unsafe_allow_html=True)
    st.markdown(
        f'''<div class="glass-panel">
            <div class="small-note"><b>Model</b><br>{MODEL_NAME}</div>
            <div style="height:0.55rem"></div>
            <div class="small-note"><b>Last category</b><br>{st.session_state.last_category or "Not available"}</div>
            <div style="height:0.55rem"></div>
            <div class="small-note"><b>Chats</b><br>{len(st.session_state.chat_sessions)}</div>
        </div>''',
        unsafe_allow_html=True
    )

chat = get_current_chat()
messages = chat["messages"]

st.markdown(
    f'''
    <div class="hero-card">
        <div class="hero-title">AI-Powered IT Project Delivery Agent</div>
        <div class="hero-subtitle">AI-powered guidance for IT project delivery — use domain &amp; topic filters, or type your query below.</div>
        <div class="metric-row">
            <div class="metric-chip">
                <div class="metric-label">Current session</div>
                <div class="metric-value">{chat["title"]}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-label">Message count</div>
                <div class="metric-value">{len(messages)}</div>
            </div>
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="chat-toolbar"><div class="chat-toolbar-title">Conversation</div><div class="chat-toolbar-title">Modern compact workspace</div></div>',
    unsafe_allow_html=True
)

for idx, message in enumerate(messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            msg_id = message.get("id", f"legacy_{idx}")
            feedback_value = message.get("feedback", None)
            st.session_state[f"feedback_{st.session_state.current_chat_id}_{msg_id}"] = feedback_value
            col_a, col_b = st.columns([0.6, 0.4])
            with col_a:
                st.feedback(
                    "thumbs",
                    key=f"feedback_{st.session_state.current_chat_id}_{msg_id}",
                    disabled=False,
                    on_change=save_feedback,
                    args=[st.session_state.current_chat_id, msg_id],
                )
            with col_b:
                render_copy_button(message["content"], key=msg_id)
            st.selectbox(
                "Store this response as context?",
                options=["Ask later", "Yes", "No"],
                index=0 if message.get("store_context") is None else (["No", "Yes"].index(message.get("store_context")) + 1),
                key=f"context_choice_{st.session_state.current_chat_id}_{msg_id}",
                on_change=save_context_choice,
                args=[st.session_state.current_chat_id, msg_id],
            )
            if st.session_state.show_debug_toggle and message.get("debug"):
                with st.expander("NLP details", expanded=False):
                    st.json(message["debug"])

st.markdown('<div class="query-card" style="border-radius:18px;padding:0.45rem 0.55rem 0.25rem 0.55rem;margin-bottom:0.38rem;">', unsafe_allow_html=True)

cols = st.columns(3)

with cols[0]:
    selected_domain = st.selectbox(
        "",
        options=DOMAINS,
        index=None,
        placeholder="Domain",
        key="selected_domain",
        label_visibility="collapsed"
    )

topic_options = DOMAIN_TOPIC_MAP.get(selected_domain, []) if selected_domain else []
role_options = DOMAIN_ROLE_MAP.get(selected_domain, []) if selected_domain else []

with cols[1]:
    selected_topic = st.selectbox(
        "",
        options=topic_options,
        index=None,
        placeholder="Quick topic",
        key="selected_topic",
        label_visibility="collapsed",
        disabled=not selected_domain
    )

with cols[2]:
    selected_role = st.selectbox(
        "",
        options=role_options,
        index=None,
        placeholder="Role (optional)",
        key="selected_role",
        label_visibility="collapsed",
        disabled=not selected_domain
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Voice-to-text section ──────────────────────────────────────────────────
#with st.expander("🎙️  Voice input (record & transcribe)", expanded=False):
#    st.markdown(
    #     '<div class="voice-badge">🎤 Powered by Gemini 1.5 Flash · Speak your query, then click Use</div>',
    #     unsafe_allow_html=True
    # )
    # audio_value = st.audio_input(
    #     "Record your query",
    #     key="voice_recorder",
    #     label_visibility="collapsed"
    # )

    # if audio_value is not None:
    #     audio_bytes = audio_value.read()
    #     audio_id = str(hash(audio_bytes))

    #     if st.session_state.get("last_processed_audio") != audio_id:
    #         with st.spinner("Transcribing audio..."):
    #             try:
    #                 transcribed = transcribe_audio_bytes(audio_bytes, mime_type="audio/wav")
    #                 if transcribed:
    #                     st.session_state.last_processed_audio = audio_id
    #                     st.session_state.voice_transcription = transcribed
    #                 else:
    #                     st.warning("Transcription returned empty. Please try speaking more clearly.")
    #             except Exception as e:
    #                 st.error(f"Transcription error: {e}")

    # transcription = st.session_state.get("voice_transcription", "")
    # if transcription:
    #     st.markdown(
    #         f'<div class="voice-transcribed">📝 {transcription}</div>',
    #         unsafe_allow_html=True
    #     )
    #     col_use, col_clear = st.columns([1, 1])
    #     with col_use:
    #         if st.button("✅  Use this transcription", use_container_width=True, key="use_transcription"):
    #             st.session_state.pending_prompt = transcription
    #             st.session_state.voice_transcription = ""
    #             st.session_state.last_processed_audio = None
    #             st.rerun()
    #     with col_clear:
    #         if st.button("🗑️  Discard", use_container_width=True, key="discard_transcription"):
    #             st.session_state.voice_transcription = ""
    #             st.session_state.last_processed_audio = None
#                st.rerun()

# ── Text chat input ────────────────────────────────────────────────────────
if st.session_state.pending_prompt:
    base_query = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
else:
    # base_query = st.chat_input("Write your query here…  (or use 🎙️ Voice input above)")
    base_query = st.chat_input("Write your query here…")

user_query = None
if base_query:
    parts = []
    if selected_domain:
        parts.append(f"Domain: {selected_domain}")
    if selected_topic:
        parts.append(f"Topic: {selected_topic}")
    if selected_role:
        parts.append(f"Role: {selected_role}")
    parts.append(base_query)
    user_query = ". ".join(parts)

if user_query:
    if chat["title"].startswith("New Chat") and len(messages) == 0:
        chat["title"] = create_chat_name(user_query)

    user_message = {"id": str(uuid.uuid4()), "role": "user", "content": user_query}
    messages.append(user_message)
    # FUTURE POSTGRESQL CODE FOR USER MESSAGE INSERTION:
    # """
    # try:
    #     conn = psycopg2.connect(...)
    #     cur = conn.cursor()
    #     cur.execute("INSERT INTO messages (message_id, chat_id, role, content, timestamp) VALUES (%s, %s, %s, %s, %s)",
    #                 (user_message["id"], st.session_state.current_chat_id, "user", user_query, datetime.now()))
    #     conn.commit()
    #     cur.close()
    #     conn.close()
    # except Exception as e:
    #     st.error(f"DB Error saving user message: {e}")
    # """

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        st.markdown(
            """
            <div class="typing-loader">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.spinner("Getting Your Query...."):
            try:
                history_messages = [m for m in messages[:-1] if m["role"] in ("user", "assistant")]
                answer, category, final_prompt, debug_info = get_model_response(
                    user_query,
                    history_messages=history_messages
                )

                st.markdown(answer)
                assistant_message = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": answer,
                    "debug": debug_info,
                    "feedback": None,
                    "store_context": None,
                }
                messages.append(assistant_message)
                # FUTURE POSTGRESQL CODE FOR ASSISTANT MESSAGE INSERTION:
                # """
                # try:
                #     conn = psycopg2.connect(...)
                #     cur = conn.cursor()
                #     cur.execute("INSERT INTO messages (message_id, chat_id, role, content, debug, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
                #                 (assistant_message["id"], st.session_state.current_chat_id, "assistant", answer, json.dumps(debug_info), datetime.now()))
                #     conn.commit()
                #     cur.close()
                #     conn.close()
                # except Exception as e:
                #     st.error(f"DB Error saving assistant message: {e}")
                # """

                col_a, col_b = st.columns([0.6, 0.4])
                with col_a:
                    st.feedback(
                        "thumbs",
                        key=f"feedback_{st.session_state.current_chat_id}_{assistant_message['id']}",
                        on_change=save_feedback,
                        args=[st.session_state.current_chat_id, assistant_message["id"]],
                    )
                with col_b:
                    render_copy_button(answer, key=assistant_message["id"])
                st.selectbox(
                    "Store this response as context?",
                    options=["Ask later", "Yes", "No"],
                    index=0,
                    key=f"context_choice_{st.session_state.current_chat_id}_{assistant_message['id']}",
                    on_change=save_context_choice,
                    args=[st.session_state.current_chat_id, assistant_message["id"]],
                )

                st.session_state.last_prompt = final_prompt
                st.session_state.last_response = answer
                st.session_state.last_category = category
                st.session_state.last_debug = debug_info

                if st.session_state.show_debug_toggle:
                    with st.expander("NLP details", expanded=False):
                        st.json(debug_info)

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                st.error(error_msg)
                messages.append({"id": str(uuid.uuid4()), "role": "assistant", "content": error_msg, "feedback": None})

if st.session_state.last_response:
    st.markdown('<div class="section-label">Output actions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            label="Download last response",
            data=st.session_state.last_response,
            file_name="it_project_agent_response.txt",
            mime="text/plain",
            use_container_width=True
        )
    with col2:
        st.info(f"Detected category: {st.session_state.last_category}")

if st.session_state.show_prompt_toggle and st.session_state.last_prompt:
    st.markdown('<div class="section-label">Prompt preview</div>', unsafe_allow_html=True)
    st.code(st.session_state.last_prompt, language="text")
