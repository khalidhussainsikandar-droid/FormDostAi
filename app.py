import os
import json
from pathlib import Path
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(
    page_title="FormSathi AI",
    page_icon="🇵🇰",
    layout="wide"
)

APP_TITLE = "FormSathi AI"
MODEL_NAME = "gemini-3.6-flash"

SYSTEM_PROMPT = """
You are FormSathi AI, a helpful multilingual form assistant for people in Pakistan.
Your job is to help users UNDERSTAND and COMPLETE forms. Do not pretend to be a government officer,
lawyer, banker, doctor, or official representative.
Always be clear, practical, and beginner-friendly. Support English, Urdu, and Roman Urdu.

For form analysis, return valid JSON with exactly these keys:
{
  "form_title": "...",
  "summary": "...",
  "fields": [
    {
      "field": "...",
      "required": true,
      "explanation": "...",
      "example": "..."
    }
  ],
  "documents": ["..."],
  "missing_or_unclear": ["..."],
  "next_steps": ["..."],
  "warnings": ["..."]
}
"""

def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing in Streamlit Secrets.")
    return genai.Client(api_key=key)

def extract_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    chunks = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            chunks.append(f"--- PAGE {i+1} ---\n{text}")
    return "\n\n".join(chunks)

def analyze_form_gemini(file_path, file_type, language):
    client = get_client()
    
    lang_instr = (
        "Answer explanations in English." if language == "English"
        else "Answer explanations in Urdu script." if language == "Urdu"
        else "Answer explanations in simple Roman Urdu."
    )
    
    prompt = f"""
{SYSTEM_PROMPT}

Analyze this form carefully. {lang_instr}
Identify fields, explanations, examples, required documents, missing info, and next steps.
Return ONLY valid JSON matching the required schema. No extra text or markdown formatting outside JSON if possible, but standard JSON is required.
"""

    contents = []
    if file_type == "pdf":
        text_content = extract_pdf_text(file_path)
        if len(text_content.strip()) > 50:
            contents = [prompt, f"\n\nFORM TEXT:\n{text_content}"]
        else:
            doc = fitz.open(file_path)
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_path = "/tmp/page1.png"
            pix.save(img_path)
            img = Image.open(img_path)
            contents = [prompt, img]
    else:
        img = Image.open(file_path)
        contents = [prompt, img]

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
    )
    return response.text

def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "", 1).replace("```", "", 1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("AI did not return valid JSON.")
    return json.loads(text[start:end + 1])

def format_analysis_ui(data: dict):
    title = data.get("form_title", "Uploaded Form")
    summary = data.get("summary", "")
    fields = data.get("fields", [])
    documents = data.get("documents", [])
    missing = data.get("missing_or_unclear", [])
    steps = data.get("next_steps", [])
    warnings = data.get("warnings", [])

    st.markdown(f"### 📄 {title}")
    st.info(f"**Summary:** {summary}")

    # Professional Tabs Layout
    tab1, tab2, tab3, tab4 = st.tabs(["🧩 Form Fields", "📋 Documents", "⚠️ Missing Info", "🧭 Next Steps"])

    with tab1:
        st.markdown("#### Detailed Field Breakdown")
        if fields:
            for item in fields:
                req = "🔴 Required" if item.get("required") else "🟢 Optional"
                with st.expander(f"{item.get('field', 'Unknown Field')} ({req})"):
                    st.markdown(f"**Explanation:** {item.get('explanation', '')}")
                    st.markdown(f"**Example to fill:** `{item.get('example', '')}`")
        else:
            st.warning("No fields detected.")

    with tab2:
        st.markdown("#### Required Documents Checklist")
        if documents:
            for doc in documents:
                st.checkbox(doc, value=False, key=f"doc_{doc}")
        else:
            st.success("No specific documents mentioned.")

    with tab3:
        st.markdown("#### Missing or Unclear Information")
        if missing:
            for item in missing:
                st.warning(item)
        else:
            st.success("Everything looks clear!")

    with tab4:
        st.markdown("#### Step-by-Step Guide")
        if steps:
            for i, step in enumerate(steps, 1):
                st.write(f"**{i}.** {step}")
        else:
            st.write("Follow official guidelines.")
            
    if warnings:
        st.error(" | ".join(warnings))

st.title("🇵🇰 FormSathi AI")
st.subheader("Samjho. Bharo. Submit Karo.")

if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("📄 Upload Form", type=["pdf", "png", "jpg", "jpeg", "webp"])
    language = st.selectbox("🌐 Language", ["Roman Urdu", "English", "Urdu"])
    analyze_btn = st.button("🧠 Analyze Form", type="primary")
    clear_btn = st.button("🧹 Clear All")

    if clear_btn:
        st.session_state.analysis_context = ""
        st.session_state.messages = []
        st.rerun()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Form Analysis")
    result_container = st.container()

    if analyze_btn:
        if not uploaded_file:
            st.error("Please upload a file first.")
        else:
            with st.spinner("Analyzing with Gemini AI..."):
                try:
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    suffix = Path(temp_path).suffix.lower()
                    file_type = "pdf" if suffix == ".pdf" else "image"
                    
                    raw = analyze_form_gemini(temp_path, file_type, language)
                    data = parse_json(raw)
                    st.session_state.analysis_context = json.dumps(data, ensure_ascii=False, indent=2)
                    st.session_state.messages = []
                    
                    with result_container:
                        format_analysis_ui(data)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        with result_container:
            if st.session_state.analysis_context:
                try:
                    format_analysis_ui(json.loads(st.session_state.analysis_context))
                except:
                    st.info("Upload a form and click Analyze.")
            else:
                st.info("Upload a form and click Analyze.")

with col2:
    st.markdown("### 💬 Ask FormSathi")
    chat_container = st.container(height=400)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if question := st.chat_input("Ask a question about the form..."):
        if not st.session_state.analysis_context:
            st.warning("Please analyze a form first.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        client = get_client()
                        chat_prompt = f"""
You are FormSathi AI. Answer the user's question based ONLY on the form analysis context provided below.
Language: {language}

FORM ANALYSIS:
{st.session_state.analysis_context}

USER QUESTION:
{question}
"""
                        response = client.models.generate_content(
                            model=MODEL_NAME,
                            contents=chat_prompt
                        )
                        answer = response.text
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
