# from transformers import BartForConditionalGeneration, BartTokenizer
# import pdfplumber

# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# def summarize_text(text):
#     custom_prompt = "Generate a concise, professional summary with clear formatting, bullet points, and key highlights from the following text:\n" + text
#     inputs = tokenizer([custom_prompt], max_length=1024, return_tensors="pt", truncation=True)
#     summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, min_length=40, length_penalty=2.0, early_stopping=True)
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# def chunk_text(text, max_words=800):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), max_words):
#         chunk = " ".join(words[i:i+max_words])
#         chunks.append(chunk)
#     return chunks

# def summarize_large_text(text):
#     chunks = chunk_text(text)
#     full_summary = ""
#     for chunk in chunks:
#         custom_prompt = (
#             "Write a detailed, well-formatted summary in bullet points covering all key information:\n\n" + chunk
#         )
#         inputs = tokenizer(
#             [custom_prompt],
#             max_length=1024,
#             return_tensors="pt",
#             truncation=True
#         )
#         summary_ids = model.generate(
#             inputs["input_ids"],
#             num_beams=4,
#             max_length=250,   # you can increase if needed
#             min_length=80,
#             length_penalty=2.0,
#             early_stopping=True
#         )
#         chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         full_summary += "\n\n" + chunk_summary
#     return full_summary.strip()

# def extract_text_from_pdf(file_path):
#     text = ""
#     with pdfplumber.open(file_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text() + "\n"
#     return text

#########################################################################
# from transformers import BartForConditionalGeneration, BartTokenizer
# import pdfplumber

# # Load model and tokenizer once globally to avoid reloading on every call
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# MAX_INPUT_TOKENS = 1024
# MAX_SUMMARY_TOKENS = 400

# def extract_text_from_pdf(file_path: str) -> str:
#     """Extract text from all pages of a PDF file."""
#     text = ""
#     with pdfplumber.open(file_path) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#     return text.strip()

# def chunk_text(text: str, max_words: int = 800) -> list[str]:
#     """Split text into chunks of maximum word count."""
#     words = text.split()
#     return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# def summarize_text(text: str) -> str:
#     """
#     Summarize a single chunk of text.
#     Uses a prompt to produce a professional, bullet-point style summary.
#     """
#     custom_prompt = (
#         "Generate a concise, professional summary with clear formatting, bullet points, "
#         "and key highlights from the following text:\n" + text
#     )
#     inputs = tokenizer(
#         [custom_prompt],
#         max_length=MAX_INPUT_TOKENS,
#         truncation=True,
#         return_tensors="pt"
#     )
#     summary_ids = model.generate(
#         inputs["input_ids"],
#         num_beams=4,
#         max_length=MAX_SUMMARY_TOKENS,
#         min_length=40,
#         length_penalty=1.0,
#         early_stopping=True,
#     )
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# def summarize_large_text(text: str) -> str:
#     """
#     Summarize large texts by chunking into manageable sizes,
#     summarizing each chunk, and concatenating results.
#     """
#     chunks = chunk_text(text)
#     summaries = []
#     for chunk in chunks:
#         chunk_summary = summarize_text(chunk)
#         summaries.append(chunk_summary)
#     return "\n\n".join(summaries).strip()

#######################
# from transformers import BartForConditionalGeneration, BartTokenizer
# import pdfplumber
# import torch
# import os

# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# MAX_INPUT_TOKENS = 1024
# MAX_SUMMARY_TOKENS = 400

# def extract_text_from_pdf(file_path: str) -> str:
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"PDF file not found at: {file_path}")
#     text = ""
#     with pdfplumber.open(file_path) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#     return text.strip() or "No extractable text found in the PDF."
# def chunk_text_by_tokens(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> list[str]:
#     words = text.split()
#     chunks = []
#     current_chunk = []
#     current_text = ""

#     for word in words:
#         current_text = " ".join(current_chunk + [word])
#         token_len = len(tokenizer.tokenize(current_text))
#         if token_len > max_tokens:
#             if current_chunk:  # Save the current chunk if it’s non-empty
#                 chunks.append(" ".join(current_chunk))
#             current_chunk = [word]  # Start new chunk with current word
#         else:
#             current_chunk.append(word)

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
#     return chunks

# def summarize_chunk(text_chunk: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> str:
#     model.to(device)
#     prompt = (
#         "Write a detailed, well-formatted summary in bullet points covering all key information:\n\n"
#         + text_chunk
#     )
#     inputs = tokenizer(
#         [prompt],
#         max_length=MAX_INPUT_TOKENS,
#         truncation=True,
#         return_tensors="pt"
#     ).to(device)
#     summary_ids = model.generate(
#         inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         num_beams=4,
#         max_length=MAX_SUMMARY_TOKENS,
#         min_length=100,
#         length_penalty=1.0,
#         early_stopping=True,
#         no_repeat_ngram_size=2,
#     )
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     # Clear memory
#     inputs = None
#     summary_ids = None
#     torch.cuda.empty_cache() if device == "cuda" else None
#     return summary.strip()

# def summarize_large_text(text: str) -> str:
#     """
#     Hierarchical summarization:
#     - Chunk input text by tokens
#     - Summarize each chunk with longer max_length
#     - Combine all chunk summaries
#     - Summarize combined summary to produce final summary
#     """
#     if not text.strip():
#         return "No text available to summarize."
#     print("Chunking input text...")
#     chunks = chunk_text_by_tokens(text)

#     print(f"Total chunks: {len(chunks)}")
#     chunk_summaries = []
#     for i, chunk in enumerate(chunks):
#         print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
#         chunk_summary = summarize_chunk(chunk)
#         chunk_summaries.append(chunk_summary)

#     combined_summary = "\n\n".join(chunk_summaries)

#     # If combined summary is very long, summarize it again
#     if len(tokenizer.tokenize(combined_summary)) > MAX_INPUT_TOKENS:
#         print("Summarizing combined summary for final output...")
#         final_summary = summarize_chunk(combined_summary)
#         return final_summary

#     return combined_summary


# text = extract_text_from_pdf(combined_summary)
# summary = summarize_large_text(text)
# print(summary)

##################

from transformers import BartForConditionalGeneration, BartTokenizer
import pdfplumber
import torch
import os
from tqdm import tqdm

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

MAX_INPUT_TOKENS = 512
MAX_SUMMARY_TOKENS = 400

def extract_text_from_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip() or "No extractable text found in the PDF."

def chunk_text_by_tokens(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> list[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_text = ""

    for word in words:
        current_text = " ".join(current_chunk + [word])
        token_len = len(tokenizer.tokenize(current_text))
        if token_len > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_chunk(text_chunk: str, no_repeat_ngram_size: int = 2) -> str:
    prompt = (
        "Write a detailed eye cathy summary , use lots of emojis, well-formatted summary in bullet points covering all key information:\n\n"
        + text_chunk
    )
    inputs = tokenizer(
        [prompt],
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    if len(tokenizer.tokenize(prompt)) > MAX_INPUT_TOKENS:
        print("Warning: Input truncated due to token limit.")
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=4,
        max_length=MAX_SUMMARY_TOKENS,
        min_length=100,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    inputs = None
    summary_ids = None
    torch.cuda.empty_cache() if device == "cuda" else None
    return summary.strip()

def summarize_large_text(text: str) -> str:
    """
    Hierarchical summarization:
    - Chunk input text by tokens
    - Summarize each chunk with longer max_length
    - Combine all chunk summaries
    - Summarize combined summary to produce final summary
    """
    if not text.strip():
        return "No text available to summarize."
    print("Chunking input text...")
    chunks = chunk_text_by_tokens(text)
    print(f"Total chunks: {len(chunks)}")
    chunk_summaries = []
    for chunk in tqdm(chunks, desc="Summarizing chunks"):
        chunk_summary = summarize_chunk(chunk)
        chunk_summaries.append(chunk_summary)

    combined_summary = "\n".join(
        line for summary in chunk_summaries for line in summary.split("\n") if line.strip()
    )

    if len(tokenizer.tokenize(combined_summary)) > MAX_INPUT_TOKENS:
        print("Summarizing combined summary for final output...")
        final_summary = summarize_chunk(combined_summary)
        return final_summary
    return combined_summary

# try:
#     text = extract_text_from_pdf("largefile.pdf")
#     summary = summarize_large_text(text)
#     print(summary)
# except Exception as e:
#     print(f"Error: {str(e)}")