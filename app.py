import os
import json
import base64
from pathlib import Path
import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

# Page Configuration
st.set_page_config(
    page_title="FormSathi AI",
    page_icon="🇵🇰",
    layout="wide"
)

APP_TITLE = "FormSathi AI"
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

SYSTEM_PROMPT = """
You are FormSathi AI, a helpful multilingual form assistant for people in Pakistan.

Your job is to help users UNDERSTAND and COMPLETE forms. You must not pretend to be a government officer,
lawyer, banker, doctor, or official representative. Do not invent official requirements, fees, deadlines,
or procedures. If information is not present in the uploaded form/context, say that it needs verification
from the official source.

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

Do not fabricate field names. Only describe fields/instructions that are visible in the uploaded form
or provided in the supplied context. If a field's meaning is unclear, say so.
"""

def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it locally as an environment variable or "
            "in Streamlit Secrets."
        )
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
    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=f"{instruction}\n\nFORM CONTEXT:\n{context[:50000]}",
    )
    return response.output_text

def call_image_model(instruction: str, path: str) -> str:
    client = get_client()
    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": instruction},
                {"type": "input_image", "image_url": image_data_url(path)},
            ],
        }],
    )
    return response.output_text

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

    md = f"## 📄 {title}\n\n"
    md += f"**Summary:** {summary}\n\n"

    md += "### 🧩 Fields\n"
    if fields:
        for item in fields:
            req = "Required" if item.get("required") else "Optional"
            md += (
                f"- **{item.get('field','Unknown field')}** (`{req}`)\n"
                f"  - *Explanation:* {item.get('explanation','')}\n"
                f"  - *Example:* {item.get('example','')}\n\n"
            )
    else:
        md += "No fields were confidently detected.\n\n"

    md += "### 📋 Required / Mentioned Documents\n"
    if documents:
        md += "\n".join([f"- ☐ {x}" for x in documents]) + "\n\n"
    else:
        md += "- No document requirement was clearly found.\n\n"

    md += "### ⚠️ Missing or Unclear\n"
    if missing:
        md += "\n".join([f"- {x}" for x in missing]) + "\n\n"
    else:
        md += "- Nothing obvious was detected.\n\n"

    md += "### 🧭 Next Steps\n"
    if steps:
        md += "\n".join([f"{i+1}. {x}" for i, x in enumerate(steps)]) + "\n\n"
    else:
        md += "No clear next steps found.\n\n"

    md += "### 🔎 Important Warnings\n"
    if warnings:
        md += "\n".join([f"- {x}" for x in warnings]) + "\n\n"
    else:
        md += "- Always verify official requirements before submitting.\n\n"

    md += "---\n> **Note:** FormSathi explains the uploaded material. It does not guarantee official approval, fees, or legal/medical advice."
    return md

# --- UI Layout ---
st.title("🇵🇰 FormSathi AI")
st.subheader("Samjho. Bharo. Submit Karo.")
st.markdown("Upload a form and let Generative AI explain its fields, identify required documents, and guide you step-by-step.")

# Initialize Session State
if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("📄 Upload Form", type=["pdf", "png", "jpg", "jpeg", "webp"])
    language = st.selectbox("🌐 Explanation Language", ["Roman Urdu", "English", "Urdu"])
    
    analyze_btn = st.button("🧠 Analyze Form", type="primary")
    clear_btn = st.button("🧹 Clear All")

    if clear_btn:
        st.session_state.analysis_context = ""
        st.session_state.messages = []
        st.rerun()

# Main Content Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Form Analysis")
    result_container = st.container()

    if analyze_btn:
        if not uploaded_file:
            st.error("Please upload a PDF or image first.")
        else:
            with st.spinner("Analyzing form with AI..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    suffix = Path(temp_path).suffix.lower()
                    lang_instr = (
                        "Answer explanations in English." if language == "English"
                        else "Answer explanations in Urdu script." if language == "Urdu"
                        else "Answer explanations in simple Roman Urdu."
                    )

                    instruction = f"""
                    Analyze this form carefully. {lang_instr}
                    Identify fields, explanations, examples, documents, missing info, and next steps.
                    Return ONLY valid JSON matching the required schema.
                    """

                    if suffix == ".pdf":
                        context = extract_pdf_text(temp_path)
                        if len(context.strip()) >= 80:
                            raw = call_text_model(instruction, context)
                        else:
                            doc = fitz.open(temp_path)
                            page = doc[0]
                            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                            temp_img = "/tmp/formsathi_page1.png"
                            pix.save(temp_img)
                            raw = call_image_model(instruction, temp_img)
                    else:
                        raw = call_image_model(instruction, temp_path)

                    data = parse_json(raw)
                    st.session_state.analysis_context = json.dumps(data, ensure_ascii=False, indent=2)
                    st.session_state.messages = [] # Reset chat on new analysis
                    
                    with result_container:
                        st.markdown(format_analysis(data))

                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        with result_container:
            if st.session_state.analysis_context:
                # Keep showing previous analysis if state exists
                try:
                    data = json.loads(st.session_state.analysis_context)
                    st.markdown(format_analysis(data))
                except:
                    st.info("Upload a form and click **Analyze Form**.")
            else:
                st.info("Upload a form and click **Analyze Form**.")

with col2:
    st.markdown("### 💬 Ask FormSathi")
    
    # Chat container
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if question := st.chat_input("Ye field mein kya likhna hai?"):
        if not st.session_state.analysis_context:
            st.warning("Please analyze a form first before asking questions.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        client = get_client()
                        lang = "English" if language == "English" else "Urdu script" if language == "Urdu" else "simple Roman Urdu"
                        prompt = f"""
                        You are answering a user about the form they uploaded. Reply in {lang}.
                        Use ONLY the form analysis context below. If not available, say so and tell them to verify officially.
                        
                        FORM ANALYSIS:
                        {st.session_state.analysis_context}
                        
                        USER QUESTION:
                        {question}
                        """
                        response = client.responses.create(
                            model=MODEL,
                            instructions=SYSTEM_PROMPT,
                            input=prompt,
                        )
                        answer = response.output_text
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        err_msg = f"❌ AI error: {e}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
