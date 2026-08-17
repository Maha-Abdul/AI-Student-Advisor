"""
AI Student Advisor
Main application entry point.

This application will allow students to:
1. Ask questions by text or voice.
2. Retrieve information from institutional catalogs and websites.
3. Generate answers only from retrieved official information.
4. Provide contact information when an answer cannot be verified.
"""

import gradio as gr


def answer_question(question):
    """
    Temporary function.

    The real RAG retrieval system will be connected here
    after we build rag.py and ingest.py.
    """
    return f"You asked: {question}"


demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Ask the AI Student Advisor",
        placeholder="Ask about programs, admissions, tuition, or student services..."
    ),
    outputs=gr.Textbox(label="Answer"),
    title="AI Student Advisor",
    description="Voice and RAG-powered institutional student information assistant."
)


if __name__ == "__main__":
    demo.launch()
