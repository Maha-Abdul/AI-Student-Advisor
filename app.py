import gradio as gr

from rag import retrieve


CONTACT_EMAIL = "admissions@example.edu"
CONTACT_PHONE = "703-555-1234"

MIN_SCORE = 0.45


def format_source(result):
    source = result["source"]
    page = result["page"]

    if page:
        return f"{source}, page {page}"

    return source


def answer_question(question):
    question = question.strip()

    if not question:
        return "Please enter a question."

    results = retrieve(question, top_k=3)

    if not results:
        return (
            "I could not find reliable information in the official resources. "
            f"Please contact us at {CONTACT_EMAIL} or {CONTACT_PHONE}."
        )

    best_result = results[0]

    if best_result["score"] < MIN_SCORE:
        return (
            "I could not verify that answer from the official resources. "
            f"Please contact us at {CONTACT_EMAIL} or {CONTACT_PHONE}."
        )

    context = best_result["text"]
    source = format_source(best_result)

    return (
        f"{context}\n\n"
        f"Source: {source}"
    )


demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Ask the AI Student Advisor",
        placeholder="Example: Do you offer an Artificial Intelligence program?"
    ),
    outputs=gr.Textbox(label="Answer"),
    title="AI Student Advisor",
    description=(
        "Ask questions about programs, admissions, tuition, "
        "student services, and institutional information."
    )
)


if __name__ == "__main__":
    demo.launch()
