import os
import json
import base64
from pathlib import Path

import gradio as gr
import fitz  # PyMuPDF
from openai import OpenAI


APP_TITLE = "FormSathi AI"
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")


SYSTEM_PROMPT = """
You are FormSathi AI, a helpful multilingual form assistant for people in Pakistan.

Your job is to help users UNDERSTAND and COMPLETE forms.

You must not pretend to be a government officer, lawyer, banker, doctor,
or official representative.

Do not invent official requirements, fees, deadlines, or procedures.

If information is not present in the uploaded form/context,
clearly say that it needs verification from the official source.

Always be clear, practical, and beginner-friendly.

Support:
- English
- Urdu
- Roman Urdu

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

Do not fabricate field names.

Only describe fields and instructions that are visible
in the uploaded form or provided context.

If a field's meaning is unclear, say so.
"""


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. "
            "Add it as an environment variable or Hugging Face Secret."
        )

    return OpenAI(api_key=api_key)


def extract_pdf_text(file_path):
    doc = fitz.open(file_path)

    pages = []

    for page_number, page in enumerate(doc):
        text = page.get_text("text")

        if text.strip():
            pages.append(
                f"--- PAGE {page_number + 1} ---\n{text}"
            )

    return "\n\n".join(pages)


def image_to_data_url(file_path):
    extension = Path(file_path).suffix.lower()

    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp"
    }

    mime_type = mime_types.get(extension, "image/png")

    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def ask_text_ai(instruction, context):
    client = get_client()

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=f"""
{instruction}

FORM CONTENT:

{context[:50000]}
"""
    )

    return response.output_text


def ask_image_ai(instruction, file_path):
    client = get_client()

    image_url = image_to_data_url(file_path)

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": instruction
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url
                    }
                ]
            }
        ]
    )

    return response.output_text


def parse_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "", 1)
        text = text.replace("```", "", 1)
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "AI did not return valid JSON."
        )

    return json.loads(
        text[start:end + 1]
    )


def format_analysis(data):
    form_title = data.get(
        "form_title",
        "Uploaded Form"
    )

    summary = data.get(
        "summary",
        ""
    )

    fields = data.get(
        "fields",
        []
    )

    documents = data.get(
        "documents",
        []
    )

    missing = data.get(
        "missing_or_unclear",
        []
    )

    next_steps = data.get(
        "next_steps",
        []
    )

    warnings = data.get(
        "warnings",
        []
    )

    output = f"""
## 📄 {form_title}

**Summary:**  
{summary}

---

## 🧩 Form Fields
"""

    if fields:

        for field in fields:

            required = (
                "Required"
                if field.get("required")
                else "Optional"
            )

            output += f"""
### {field.get("field", "Unknown Field")}

**Status:** `{required}`

**Explanation:**  
{field.get("explanation", "")}

**Example:**  
{field.get("example", "")}

"""

    else:

        output += """
No fields were confidently detected.
"""

    output += """
---

## 📋 Required / Mentioned Documents
"""

    if documents:

        for document in documents:
            output += f"- ☐ {document}\n"

    else:

        output += (
            "- No document requirement was clearly found.\n"
        )

    output += """
---

## ⚠️ Missing or Unclear Information
"""

    if missing:

        for item in missing:
            output += f"- {item}\n"

    else:

        output += "- Nothing obvious was detected.\n"

    output += """
---

## 🧭 Next Steps
"""

    if next_steps:

        for index, step in enumerate(
            next_steps,
            start=1
        ):
            output += f"{index}. {step}\n"

    else:

        output += (
            "No clear next steps were found.\n"
        )

    output += """
---

## 🔎 Important Warnings
"""

    if warnings:

        for warning in warnings:
            output += f"- {warning}\n"

    else:

        output += (
            "- Verify final requirements with "
            "the official organization.\n"
        )

    output += """

---

> ⚠️ FormSathi AI explains uploaded material.
> It is not an official government service and
> does not guarantee eligibility, approval, fees,
> deadlines, or legal/medical advice.
"""

    return output


def analyze_form(file_path, language):

    if not file_path:

        return (
            "❌ Please upload a form first.",
            ""
        )

    try:

        language_instruction = {
            "English":
                "Explain everything in simple English.",

            "Urdu":
                "Explain everything in simple Urdu script.",

            "Roman Urdu":
                "Explain everything in simple Roman Urdu."
        }

        instruction = f"""
Analyze this form carefully.

{language_instruction[language]}

Identify only the fields and instructions
that you can actually see or extract.

For every field:

- Explain what the user should enter.
- Tell whether it appears required or optional.
- Give a simple example when appropriate.

Also identify:

- Documents mentioned in the form.
- Missing or unclear information.
- Next steps based only on the uploaded material.
- Important warnings.

Return ONLY valid JSON using the required schema.
"""

        extension = Path(file_path).suffix.lower()

        if extension == ".pdf":

            text = extract_pdf_text(
                file_path
            )

            if len(text.strip()) >= 80:

                raw_response = ask_text_ai(
                    instruction,
                    text
                )

            else:

                # Scanned PDF fallback
                document = fitz.open(file_path)

                page = document[0]

                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(
                        1.5,
                        1.5
                    ),
                    alpha=False
                )

                temp_file = "/tmp/form_page.png"

                pixmap.save(temp_file)

                raw_response = ask_image_ai(
                    instruction,
                    temp_file
                )

        elif extension in [
            ".png",
            ".jpg",
            ".jpeg",
            ".webp"
        ]:

            raw_response = ask_image_ai(
                instruction,
                file_path
            )

        else:

            return (
                "❌ Unsupported file type.",
                ""
            )

        result = parse_json(
            raw_response
        )

        analysis_context = json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )

        formatted_result = format_analysis(
            result
        )

        return (
            formatted_result,
            analysis_context
        )

    except Exception as error:

        return (
            f"❌ Error: {error}",
            ""
        )


def ask_ai(
    question,
    chat_history,
    analysis_context,
    language
):

    if not question.strip():

        return (
            chat_history,
            ""
        )

    if not analysis_context:

        answer = (
            "Please upload and analyze a form first."
        )

    else:

        try:

            client = get_client()

            language_instruction = {
                "English":
                    "Answer in simple English.",

                "Urdu":
                    "Answer in simple Urdu script.",

                "Roman Urdu":
                    "Answer in simple Roman Urdu."
            }

            prompt = f"""
You are helping the user understand
their uploaded form.

{language_instruction[language]}

Use ONLY the form analysis below.

If the answer is not available,
say that clearly.

FORM ANALYSIS:

{analysis_context}

USER QUESTION:

{question}
"""

            response = client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=prompt
            )

            answer = response.output_text

        except Exception as error:

            answer = f"❌ AI Error: {error}"

    if chat_history is None:
        chat_history = []

    chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return (
        chat_history,
        ""
    )


def clear_all():

    return (
        None,
        "",
        [],
        "",
        ""
    )


# =========================
# GRADIO USER INTERFACE
# =========================

with gr.Blocks(
    title=APP_TITLE,
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
# 🇵🇰 FormSathi AI

### **Samjho. Bharo. Submit Karo.**

Upload a complicated form and let Generative AI:

- 🧠 Understand the form
- 📄 Explain each field
- 🇵🇰 Explain in Urdu / Roman Urdu
- 📋 Identify required documents
- ⚠️ Find missing or unclear information
- 💬 Answer questions about the form

> **Demo MVP:** Always verify final requirements
> with the official organization.
"""
    )

    analysis_context = gr.State("")

    with gr.Row():

        with gr.Column(
            scale=1
        ):

            file_input = gr.File(
                label="📄 Upload Your Form",
                file_types=[
                    ".pdf",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp"
                ],
                type="filepath"
            )

            language = gr.Dropdown(
                choices=[
                    "English",
                    "Urdu",
                    "Roman Urdu"
                ],
                value="Roman Urdu",
                label="🌐 Explanation Language"
            )

            analyze_button = gr.Button(
                "🧠 Analyze Form",
                variant="primary"
            )

            clear_button = gr.Button(
                "🧹 Clear"
            )

        with gr.Column(
            scale=2
        ):

            result = gr.Markdown(
                """
Upload a form and click
**Analyze Form**.
"""
            )

    gr.Markdown("---")

    gr.Markdown(
        """
## 💬 Ask FormSathi
"""
    )

    chatbot = gr.Chatbot(
        label="Form Assistant",
        type="messages",
        height=350
    )

    with gr.Row():

        question = gr.Textbox(
            label="Ask your question",
            placeholder=(
                "Example: Ye field mein kya likhna hai?"
            ),
            scale=5
        )

        ask_button = gr.Button(
            "Ask AI",
            variant="primary",
            scale=1
        )

    analyze_button.click(
        fn=analyze_form,
        inputs=[
            file_input,
            language
        ],
        outputs=[
            result,
            analysis_context
        ]
    )

    ask_button.click(
        fn=ask_ai,
        inputs=[
            question,
            chatbot,
            analysis_context,
            language
        ],
        outputs=[
            chatbot,
            question
        ]
    )

    question.submit(
        fn=ask_ai,
        inputs=[
            question,
            chatbot,
            analysis_context,
            language
        ],
        outputs=[
            chatbot,
            question
        ]
    )

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            file_input,
            result,
            chatbot,
            analysis_context,
            question
        ]
    )


if __name__ == "__main__":
    demo.launch()
