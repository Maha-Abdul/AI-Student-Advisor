import asyncio
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import edge_tts
import gradio as gr
import whisper
from openai import OpenAI

from rag import build_context, retrieve


CONTACT_EMAIL = "info@NATI.edu"
CONTACT_PHONE = "833-228-1010"

MIN_SCORE = 0.40

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")


@lru_cache(maxsize=1)
def get_whisper_model():
    return whisper.load_model(WHISPER_MODEL)


def transcribe_audio(audio_path):
    if not audio_path:
        return ""

    result = get_whisper_model().transcribe(
        str(audio_path),
        fp16=False
    )

    return result.get("text", "").strip()


def format_source(result):
    source = result.get("source") or "NATI official resource"
    page = result.get("page")
    url = result.get("url")

    parts = [str(source)]

    if page is not None and page != "":
        parts.append(f"page {page}")

    if url:
        parts.append(str(url))

    return " | ".join(parts)


def format_references(results):
    references = []

    for result in results:
        reference = format_source(result)

        if reference not in references:
            references.append(reference)

    return "\n".join(
        f"{number}. {reference}"
        for number, reference in enumerate(references, start=1)
    )


def grounded_answer(question, results):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add it as a Hugging Face Space secret."
        )

    context = build_context(results)

    client = OpenAI()

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are the NATI AI Student Advisor. "
            "Answer using only the NATI context supplied to you. "
            "Do not use outside knowledge. "
            "Never invent facts. "
            "Give a direct, natural answer in no more than three short sentences. "
            "Do not include citations, URLs, source labels, copyright notices, "
            "page numbers, headings, or contact details in the answer. "
            "If the supplied context does not support the answer, say exactly: "
            "'I could not verify that from NATI's official resources.'"
        ),
        input=(
            f"QUESTION:\n{question}\n\n"
            f"NATI CONTEXT:\n{context}"
        ),
        max_output_tokens=180,
    )

    answer = response.output_text.strip()

    if not answer:
        raise RuntimeError("OpenAI returned an empty answer.")

    return answer


def is_small_talk(question):
    text = question.lower().strip()

    small_talk_phrases = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how are you?",
        "thanks",
        "thank you",
        "thank you very much",
        "bye",
        "goodbye",
        "see you",
        "what's up",
        "whats up"
    ]

    return text in small_talk_phrases


def small_talk_answer(question):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add it as a Hugging Face Space secret."
        )

    client = OpenAI()

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are the NATI AI Student Advisor. "
            "This is casual conversation, not a document search. "
            "Respond naturally, warmly, and briefly. "
            "Use no more than two short sentences. "
            "If appropriate, invite the user to ask a question about NATI. "
            "Do not include sources or references."
        ),
        input=question,
        max_output_tokens=80,
    )

    answer = response.output_text.strip()

    if not answer:
        return "Hi! How can I help you with NATI today?"

    return answer


async def create_speech_file(answer):
    with tempfile.NamedTemporaryFile(
        prefix="nati_answer_",
        suffix=".mp3",
        delete=False
    ) as file:
        output_path = Path(file.name)

    communicator = edge_tts.Communicate(
        answer,
        TTS_VOICE
    )

    await communicator.save(str(output_path))

    return str(output_path)


def text_to_speech(answer):
    return asyncio.run(
        create_speech_file(answer)
    )


def answer_question(audio_path, typed_question):
    try:
        if audio_path:
            question = transcribe_audio(audio_path)
        else:
            question = (typed_question or "").strip()

    except Exception as error:
        return (
            "",
            f"I could not understand the recording: {error}",
            "",
            None
        )

    if not question:
        return (
            "",
            "Please record a question or type one below.",
            "",
            None
        )

    # -----------------------------------------------------
    # SMALL TALK
    # -----------------------------------------------------

    if is_small_talk(question):
        try:
            answer = small_talk_answer(question)
        except Exception as error:
            answer = f"Hi! How can I help you with NATI today?"

        try:
            audio_output = text_to_speech(answer)
        except Exception:
            audio_output = None

        return (
            question,
            answer,
            "",
            audio_output
        )

    # -----------------------------------------------------
    # NATI RAG RETRIEVAL
    # -----------------------------------------------------

    try:
        results = retrieve(
            question,
            top_k=3
        )

    except Exception as error:
        return (
            question,
            f"The NATI retrieval system encountered an error: {error}",
            "",
            None
        )

    if not results or results[0].get("score", 0) < MIN_SCORE:
        answer = (
            "I could not verify that from NATI's official resources. "
            f"Please contact NATI at {CONTACT_EMAIL} or {CONTACT_PHONE}."
        )

        try:
            audio_output = text_to_speech(answer)
        except Exception:
            audio_output = None

        return (
            question,
            answer,
            "No sufficiently relevant NATI source was found.",
            audio_output
        )

    references = format_references(results)

    try:
        answer = grounded_answer(
            question,
            results
        )

    except Exception as error:
        return (
            question,
            f"I could not create the answer: {error}",
            references,
            None
        )

    try:
        audio_output = text_to_speech(answer)

    except Exception as error:
        return (
            question,
            answer,
            references + f"\n\nVoice error: {error}",
            None
        )

    return (
        question,
        answer,
        references,
        audio_output
    )


with gr.Blocks(title="NATI AI Student Advisor") as demo:

    gr.Markdown(
        "# NATI AI Student Advisor\n\n"
        "Ask a question by voice or text.\n\n"
        "For voice questions, record your question and stop recording. "
        "The advisor will automatically process it and respond."
    )

    with gr.Row():

        microphone = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak your question"
        )

        typed_question = gr.Textbox(
            label="Or type your question",
            placeholder="Example: What are the admission requirements?",
            lines=3
        )

    transcript = gr.Textbox(
        label="What You Said",
        interactive=False
    )

    answer = gr.Textbox(
        label="Answer",
        lines=5,
        interactive=False
    )

    references = gr.Textbox(
        label="Source / Reference",
        lines=4,
        interactive=False
    )

    spoken_answer = gr.Audio(
        label="Spoken Answer",
        autoplay=True,
        interactive=False
    )

    microphone.stop_recording(
        fn=answer_question,
        inputs=[
            microphone,
            typed_question
        ],
        outputs=[
            transcript,
            answer,
            references,
            spoken_answer
        ]
    )

    typed_question.submit(
        fn=answer_question,
        inputs=[
            microphone,
            typed_question
        ],
        outputs=[
            transcript,
            answer,
            references,
            spoken_answer
        ]
    )


if __name__ == "__main__":
    demo.launch()
