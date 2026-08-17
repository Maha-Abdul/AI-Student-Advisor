# AI Student Advisor

An AI-powered voice and Retrieval-Augmented Generation (RAG) assistant designed to answer student questions using official institutional catalogs and website content.

#Project Objective

The goal of this project is to provide students with fast and reliable answers about:

- Academic programs
- Admissions requirements
- Tuition and fees
- Program duration
- Student services
- Institutional policies

The assistant retrieves information from approved institutional sources before generating an answer.

If the system cannot find enough reliable information, it does not guess. Instead, it directs the student to the institution's official email or phone number.

## Planned System Architecture

Student Voice or Text  
↓  
Speech-to-Text  
↓  
Question Processing  
↓  
RAG Retrieval  
↓  
Institutional Catalog + Website  
↓  
Large Language Model  
↓  
Grounded Answer with Source  
↓  
Text and Voice Response

# Technologies

- Python
- Retrieval-Augmented Generation (RAG)
- FAISS
- Sentence Transformers
- Whisper
- Large Language Models
- Gradio
- Hugging Face

# Key Features

- Voice and text questions
- Retrieval from institutional documents
- Website knowledge retrieval
- Source-grounded answers
- Hallucination reduction
- Safe fallback to email or phone
- Live web interface

# Project Status

Under development.
