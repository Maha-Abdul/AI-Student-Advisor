import asyncio
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import edge_tts
import gradio as gr
import torch
import whisper
from openai import OpenAI
from sentence_transformers import CrossEncoder

from rag import build_context, retrieve


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

CONTACT_EMAIL = "info@NATI.edu"
CONTACT_PHONE = "833-228-1010"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")

# Retrieve more candidates first
RETRIEVAL_TOP_K = 10

# After reranking, only send strongest passages to OpenAI
RERANK_TOP_K = 4

# Minimum acceptable reranker confidence
MIN_RERANK_SCORE = 0.18


# ---------------------------------------------------------
# OPENAI CLIENT
# ---------------------------------------------------------

def get_openai_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add it as a Hugging Face Space secret."
        )

    return OpenAI()


# ---------------------------------------------------------
# WHISPER
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# CROSS ENCODER RERANKER
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        activation_fn=torch.nn.Sigmoid()
    )


def rerank_results(question, results):
    if not results:
        return []

    pairs = [
        [question, result.get("text", "")]
        for result in results
    ]

    scores = get_reranker().predict(pairs)

    ranked = []

    for result, score in zip(results, scores):
        item = dict(result)
        item["rerank_score"] = float(score)
        ranked.append(item)

    ranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked


# ---------------------------------------------------------
# SOURCE FORMATTING
# ---------------------------------------------------------

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
        f"- {reference}"
        for reference in references
    )


# ---------------------------------------------------------
# SMALL TALK
# ---------------------------------------------------------

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_small_talk(question):
    text = normalize_text(question)

    phrases = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how are you doing",
        "thanks",
        "thank you",
        "thank you very much",
        "bye",
        "goodbye",
        "see you",
        "what's up",
        "whats up"
    ]

    return text in phrases


def small_talk_answer(question, memory):
    client = get_openai_client()

    recent_memory = memory[-6:] if memory else []

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are the NATI AI Student Advisor. "
            "Respond naturally and briefly to greetings, thanks, "
            "goodbyes, or casual conversation. "
            "Use no more than two short sentences. "
            "Do not include sources. "
            "When appropriate, invite the user to ask about NATI."
        ),
        input=(
            f"Recent conversation:\n{recent_memory}\n\n"
            f"User:\n{question}"
        ),
        max_output_tokens=80
    )

    answer = response.output_text.strip()

    if not answer:
        return "Hi! How can I help you with NATI today?"

    return answer


# ---------------------------------------------------------
# SEMANTIC QUERY REWRITING
# ---------------------------------------------------------

def build_semantic_query(question, memory):
    client = get_openai_client()

    recent_memory = memory[-6:] if memory else []

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "Rewrite the user's message into one short standalone search query "
            "for NATI's official catalog and website. "
            "Preserve the user's intended meaning, not just their exact wording. "
            "Resolve synonyms and related meanings. "
            "Examples: "
            "'where are you located', 'campus location', 'address', and "
            "'where is NATI' all mean NATI campus/contact address. "
            "'cost', 'price', and 'how much' may mean tuition or fees depending "
            "on conversation context. "
            "Use conversation history only to resolve references such as "
            "'that program', 'it', 'that course', or 'there'. "
            "Do not answer the question. "
            "Return only the rewritten search query."
        ),
        input=(
            f"Conversation history:\n{recent_memory}\n\n"
            f"Latest user message:\n{question}"
        ),
        max_output_tokens=80
    )

    rewritten = response.output_text.strip()

    return rewritten if rewritten else question


# ---------------------------------------------------------
# EVIDENCE VALIDATION
# ---------------------------------------------------------

def evidence_is_strong(ranked_results):
    if not ranked_results:
        return False

    best_score = ranked_results[0].get("rerank_score", 0)

    return best_score >= MIN_RERANK_SCORE


# ---------------------------------------------------------
# GROUNDED ANSWER
# ---------------------------------------------------------

def grounded_answer(question, results, memory):
    client = get_openai_client()

    context = build_context(results)
    recent_memory = memory[-6:] if memory else []

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are the NATI AI Student Advisor. "
            "Answer factual NATI questions using only the supplied NATI context. "
            "Conversation history may only be used to understand what the user "
            "is referring to. "
            "Do not use outside knowledge. "
            "Never invent, infer, or guess facts that are not explicitly supported "
            "by the NATI context. "
            "For addresses, campus locations, phone numbers, tuition amounts, "
            "dates, program names, accreditation, policies, and other exact facts, "
            "only state them if they appear clearly in the supplied context. "
            "If multiple sources conflict, do not choose one silently. "
            "Say that the information could not be verified. "
            "Give a direct, natural answer in no more than three short sentences. "
            "Do not include source labels, URLs, page numbers, citations, "
            "copyright notices, or headings in the answer. "
            "If the context does not clearly support the answer, say exactly: "
            "'I could not verify that from NATI's official resources.'"
        ),
        input=(
            f"Conversation history:\n{recent_memory}\n\n"
            f"User question:\n{question}\n\n"
            f"NATI CONTEXT:\n{context}"
        ),
        max_output_tokens=180
    )

    answer = response.output_text.strip()

    if not answer:
        raise RuntimeError("OpenAI returned an empty answer.")

    return answer


# ---------------------------------------------------------
# TEXT TO SPEECH
# ---------------------------------------------------------

async def create_speech_file(answer):
    with tempfile.NamedTemporaryFile(
        prefix="nati_answer_",
        suffix=".mp3",
        delete=False
    ) as file:
        output_path = Path(file.name)

    communicator = edge_tts.Communicate(
        answer,
        TTS_VOICE,
        rate="-3%"
    )

    await communicator.save(str(output_path))

    return str(output_path)


def text_to_speech(answer):
    return asyncio.run(
        create_speech_file(answer)
    )


# ---------------------------------------------------------
# MAIN CONVERSATION ENGINE
# ---------------------------------------------------------

def process_question(question, chat_history, memory):
    chat_history = chat_history or []
    memory = memory or []

    question = (question or "").strip()

    if not question:
        return chat_history, memory, None

    # Add user message
    chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    memory.append(
        {
            "role": "user",
            "content": question
        }
    )

    # -----------------------------------------------------
    # SMALL TALK
    # -----------------------------------------------------

    if is_small_talk(question):

        try:
            answer = small_talk_answer(
                question,
                memory
            )
        except Exception:
            answer = "Hi! How can I help you with NATI today?"

        chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        memory.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        try:
            audio_output = text_to_speech(answer)
        except Exception:
            audio_output = None

        return (
            chat_history,
            memory,
            audio_output
        )

    # -----------------------------------------------------
    # SEMANTIC QUERY
    # -----------------------------------------------------

    try:
        search_query = build_semantic_query(
            question,
            memory
        )
    except Exception:
        search_query = question

    # -----------------------------------------------------
    # RETRIEVE MORE CANDIDATES
    # -----------------------------------------------------

    try:
        raw_results = retrieve(
            search_query,
            top_k=RETRIEVAL_TOP_K
        )

    except Exception:

        answer = (
            "I encountered a problem while searching NATI's resources."
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        memory.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        return (
            chat_history,
            memory,
            None
        )

    # -----------------------------------------------------
    # SEMANTIC RERANKING
    # -----------------------------------------------------

    try:
        ranked_results = rerank_results(
            search_query,
            raw_results
        )
    except Exception:
        ranked_results = raw_results

    if not evidence_is_strong(ranked_results):

        answer = (
            "I could not verify that from NATI's official resources. "
            f"Please contact NATI at {CONTACT_EMAIL} "
            f"or {CONTACT_PHONE}."
        )

        chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        memory.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        try:
            audio_output = text_to_speech(answer)
        except Exception:
            audio_output = None

        return (
            chat_history,
            memory,
            audio_output
        )

    best_results = ranked_results[:RERANK_TOP_K]

    # -----------------------------------------------------
    # GENERATE GROUNDED ANSWER
    # -----------------------------------------------------

    try:
        answer = grounded_answer(
            question,
            best_results,
            memory
        )

    except Exception:
        answer = "I could not generate the answer right now."

    references = format_references(
        best_results
    )

    display_answer = (
        f"{answer}\n\n"
        f"**Source:**\n"
        f"{references}"
    )

    chat_history.append(
        {
            "role": "assistant",
            "content": display_answer
        }
    )

    memory.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # -----------------------------------------------------
    # SPEAK ANSWER ONLY
    # -----------------------------------------------------

    try:
        audio_output = text_to_speech(answer)

    except Exception:
        audio_output = None

    return (
        chat_history,
        memory,
        audio_output
    )


# ---------------------------------------------------------
# TEXT HANDLER
# ---------------------------------------------------------

def handle_text(message, chat_history, memory):
    updated_chat, updated_memory, audio_output = process_question(
        message,
        chat_history,
        memory
    )

    return (
        "",
        updated_chat,
        updated_memory,
        audio_output
    )


# ---------------------------------------------------------
# VOICE HANDLER
# ---------------------------------------------------------

def handle_voice(audio_path, chat_history, memory):
    if not audio_path:
        return (
            chat_history,
            memory,
            None,
            None
        )

    try:
        question = transcribe_audio(audio_path)
    except Exception:
        question = ""

    updated_chat, updated_memory, audio_output = process_question(
        question,
        chat_history,
        memory
    )

    return (
        updated_chat,
        updated_memory,
        audio_output,
        None
    )


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

with gr.Blocks(
    title="NATI AI Student Advisor"
) as demo:

    gr.Markdown(
        """
# 🎓 NATI AI Student Advisor

Ask naturally by **voice or text**.

🎤 **Record → Speak → Stop → Automatic AI Response**

Answers are grounded in NATI's official catalog and website.
"""
    )

    memory_state = gr.State([])

    chatbot = gr.Chatbot(
        value=[
            {
                "role": "assistant",
                "content": "Hi! How can I help you with NATI today?"
            }
        ],
        height=500
    )

    text_input = gr.Textbox(
        placeholder="Type a message and press Enter...",
        label="Message",
        lines=1
    )

    microphone = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="🎤 Speak"
    )

    spoken_answer = gr.Audio(
        label="Voice Response",
        autoplay=True,
        interactive=False
    )

    text_input.submit(
        fn=handle_text,
        inputs=[
            text_input,
            chatbot,
            memory_state
        ],
        outputs=[
            text_input,
            chatbot,
            memory_state,
            spoken_answer
        ]
    )

    microphone.stop_recording(
        fn=handle_voice,
        inputs=[
            microphone,
            chatbot,
            memory_state
        ],
        outputs=[
            chatbot,
            memory_state,
            spoken_answer,
            microphone
        ]
    )


if __name__ == "__main__":
    demo.launch()
