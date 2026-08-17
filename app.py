import os
import gradio as gr
from openai import OpenAI

from rag import retrieve, build_context


CONTACT_EMAIL = "info@NATI.edu"
CONTACT_PHONE = "833-228-1010"

MIN_SCORE = 0.40

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def format_sources(results):
    sources = []

    for result in results:
        if result.get("url"):
            source = result["url"]

        elif result.get("page"):
            source = f"{result['source']}, page {result['page']}"

        else:
            source = result["source"]

        if source not in sources:
            sources.append(source)

    return "\n".join(
        f"- {source}"
        for source in sources[:3]
    )


def answer_question(question):

    question = question.strip()

    if not question:
        return "Please enter a question."

    # Step 1: Retrieve relevant NATI information
    results = retrieve(
        question,
        top_k=5
    )

    # Step 2: No useful retrieval
    if not results or results[0]["score"] < MIN_SCORE:
        return (
            "I could not verify that information from NATI's "
            "official catalog or website.\n\n"
            f"Please contact NATI:\n"
            f"Email: {CONTACT_EMAIL}\n"
            f"Phone: {CONTACT_PHONE}"
        )

    # Step 3: Build RAG context
    context = build_context(results)

    instructions = f"""
You are the AI Student Advisor for North America Technical Institute (NATI).

Your job is to answer student questions using ONLY the official NATI
information provided in the retrieved context.

RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not guess.
4. Do not invent tuition, fees, program lengths, schedules,
   admission requirements, accreditation information, policies,
   visa information, or financial aid information.
5. If the provided context does not contain enough information
   to answer the student's question, say that you cannot verify
   the answer.
6. Keep the answer clear, friendly, and concise.
7. Do not claim that information is current unless the source says so.
8. Do not make promises on behalf of NATI.
9. If the question cannot be answered reliably, direct the student to:

Email: {CONTACT_EMAIL}
Phone: {CONTACT_PHONE}

RETRIEVED NATI CONTEXT:

{context}
"""

    # Step 4: Generate grounded answer
    response = client.responses.create(
        model="gpt-5.6",
        instructions=instructions,
        input=question,
    )

    answer = response.output_text.strip()

    # Step 5: Show sources
    sources = format_sources(results)

    return (
        f"{answer}\n\n"
        f"Sources:\n{sources}"
    )


demo = gr.Interface(
    fn=answer_question,

    inputs=gr.Textbox(
        label="Ask the NATI AI Student Advisor",
        placeholder=(
            "Example: What are the admission requirements?"
        ),
        lines=2,
    ),

    outputs=gr.Textbox(
        label="Answer",
        lines=12,
    ),

    title="NATI AI Student Advisor",

    description=(
        "Ask questions about NATI programs, admissions, tuition, "
        "policies, student services, and other institutional information. "
        "Answers are grounded in NATI's official catalog and website."
    ),

    examples=[
        ["What are the admission requirements?"],
        ["How can I register for classes?"],
        ["What programs does NATI offer?"],
        ["What is NATI's refund policy?"],
        ["How do I contact NATI?"],
    ],
)


if __name__ == "__main__":
    demo.launch()
