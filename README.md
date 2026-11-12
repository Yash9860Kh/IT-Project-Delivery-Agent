# AI-Powered IT Project Delivery Agent

This project is a Streamlit-based AI assistant that helps answer IT project delivery queries using prompt engineering, IT standards, and lightweight NLP techniques.

## Features
- Query preprocessing
- Keyword extraction
- Intent and category detection
- Prompt template mapping
- Gemini API based response generation
- Streamlit chat interface with NLP details

## Project Structure
- `app.py` - Streamlit user interface
- `src/main.py` - Main workflow logic
- `src/nlp_utils.py` - NLP preprocessing and category detection
- `src/prompt_loader.py` - Prompt and file loading functions
- `src/response_generator.py` - Gemini API response generation
- `prompts/prompt_library.md` - Detailed prompt templates
- `data/standards/it_standards.md` - IT standards reference
- `data/samples/test_queries.md` - Sample queries for testing

## Setup
1. Create a `.env` file in the project root.
2. Add the line: `GEMINI_API_KEY=your_api_key_here`
3. Install dependencies:
   `pip install -r requirements.txt`
4. Run the application:
   `streamlit run app.py`

## Purpose
This project supports academic and practical IT project guidance by combining prompt engineering with basic NLP and generative AI response generation.

# PROJECT FOLDER STRUCTURE
AI_Project_Delivery_Agent/
│
├── data/
│   │
│   ├── samples/
│   │   └── test_queries.md
│   │
│   └── standards/
│       └── it_standards.md
│
├── prompts/
│   └── prompt_library.md
│
├── src/
│   ├── main.py
│   └── nlp_utils.py
│
├── tests/
│
├── .env
├── app.py
├── README.md
└── requirements.txt