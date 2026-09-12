import os
import json
import base64
from pathlib import Path
import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

st.set_page_config(
    page_title="FormSathi AI",
    page_icon="🇵🇰",
    layout="wide"
)

APP_TITLE = "FormSathi AI"
MODEL = os.getenv("OPENAI_MODEL", "llama-3.2-90b-vision-preview")

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
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("API_KEY is missing in Streamlit Secrets.")
    
    # Yeh line zaroori hai agar aap Groq key ('gsk_') use kar rahe hain
    if key.startswith("gsk_"):
        return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    
    return OpenAI(api_key=key)

def extract_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    chunks = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            chunks.append(f"--- PAGE {i+1} ---\n{text}")
    return "\n\n".join(chunks)

def image_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    data = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"

def call_text_model(instruction: str, context: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\nFORM CONTEXT:\n{context[:30000]}"}
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def call_image_model(instruction: str, path: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": image_data_url(path)}}
                ]
            }
        ],
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "", 1).replace("```", "", 1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("AI did not return valid JSON.")
    return json.loads(text[start:end + 1])

def format_analysis(data: dict) -> str:
    title = data.get("form_title", "Uploaded Form")
    summary = data.get("summary", "")
    fields = data.get("fields", [])
    documents = data.get("documents", [])
    missing = data.get("missing_or_unclear", [])
    steps = data.get("next_steps", [])
    warnings = data.get("warnings", [])

    md = f"## 📄 {title}\n\n**Summary:** {summary}\n\n### 🧩 Fields\n"
    if fields:
        for item in fields:
            req = "Required" if item.get("required") else "Optional"
            md += f"- **{item.get('field','Unknown')}** (`{req}`)\n  - *Explanation:* {item.get('explanation','')}\n  - *Example:* {item.get('example','')}\n\n"
    else:
        md += "No fields detected.\n\n"

    md += "### 📋 Required Documents\n"
    md += "\n".join([f"- ☐ {x}" for x in documents]) + "\n\n" if documents else "- None found.\n\n"
    md += "### ⚠️ Missing / Unclear\n"
    md += "\n".join([f"- {x}" for x in missing]) + "\n\n" if missing else "- None.\n\n"
    md += "### 🧭 Next Steps\n"
    md += "\n".join([f"{i+1}. {x}" for i, x in enumerate(steps)]) + "\n\n" if steps else "- Verify officially.\n\n"
    return md

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
            with st.spinner("Analyzing with AI..."):
                try:
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    suffix = Path(temp_path).suffix.lower()
                    lang_instr = "Answer in Roman Urdu." if language == "Roman Urdu" else "Answer in English."
                    instruction = f"Analyze this form. {lang_instr} Return ONLY valid JSON."

                    if suffix == ".pdf":
                        context = extract_pdf_text(temp_path)
                        raw = call_text_model(instruction, context)
                    else:
                        raw = call_image_model(instruction, temp_path)

                    data = parse_json(raw)
                    st.session_state.analysis_context = json.dumps(data, ensure_ascii=False, indent=2)
                    st.session_state.messages = []
                    
                    with result_container:
                        st.markdown(format_analysis(data))
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        with result_container:
            if st.session_state.analysis_context:
                try:
                    st.markdown(format_analysis(json.loads(st.session_state.analysis_context)))
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
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": f"Context:\n{st.session_state.analysis_context}\n\nQuestion:\n{question}"}
                            ]
                        )
                        answer = response.choices[0].message.content
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
