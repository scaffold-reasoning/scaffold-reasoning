from typing import List
import logging
import fitz  # PyMuPDF
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_pdf(pdf_path: str) -> str:
    """
    Read text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = []
        with fitz.open(pdf_path) as doc:
            logger.info(f"Processing PDF with {len(doc)} pages")
            for page in doc:
                text.append(page.get_text())
        
        return "\n".join(filter(None, text))  # Filter out empty strings
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise


def split_into_chunks(text: str, chunk_token_size: int, model_name: str) -> List[str]:
    """
    Split text into chunks based on token count, preserving paragraph boundaries when possible.
    
    Args:
        text: The input text to split.
        chunk_token_size: Maximum number of tokens per chunk.
        tokenizer_name: Name of the tiktoken encoding to use (default: "cl100k_base").
    
    Returns:
        List[str]: List of text chunks, each within chunk_token_size tokens.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    """chunked_context = []

    for subject, details in context:
        chunk = {
            "subject": subject,
            "content": " ".join(details)  # or "\n".join(details) for clearer formatting
        }
    chunked_context.append(chunk)"""

    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk_tokens = []
    current_chunk_words = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        
        # Tokenize paragraph
        paragraph_tokens = tokenizer.encode(paragraph,  disallowed_special=())
        
        # If paragraph alone is larger than chunk_token_size, split paragraph forcibly by tokens
        if len(paragraph_tokens) > chunk_token_size:
            # Flush current chunk first if any
            if current_chunk_words:
                chunks.append(' '.join(current_chunk_words))
                current_chunk_words = []
                current_chunk_tokens = []
            
            # Split large paragraph into token-size chunks
            start_idx = 0
            while start_idx < len(paragraph_tokens):
                end_idx = start_idx + chunk_token_size
                sub_chunk_tokens = paragraph_tokens[start_idx:end_idx]
                sub_chunk_text = tokenizer.decode(sub_chunk_tokens)
                chunks.append(sub_chunk_text)
                start_idx = end_idx
            continue
        
        # If adding paragraph tokens exceeds chunk size, flush current chunk and start new
        if len(current_chunk_tokens) + len(paragraph_tokens) > chunk_token_size:
            if current_chunk_words:
                chunks.append(' '.join(current_chunk_words))
            current_chunk_words = [paragraph]
            current_chunk_tokens = paragraph_tokens
        else:
            # Add paragraph to current chunk
            current_chunk_words.append(paragraph)
            current_chunk_tokens.extend(paragraph_tokens)
    
    # Flush remaining chunk if any
    if current_chunk_words:
        chunks.append(' '.join(current_chunk_words))
    
    print(f"Token-based split: {len(chunks)} chunks of max {chunk_token_size} tokens.")
    for chunk in chunks:
        print(f"Chunk size: {len(tokenizer.encode(chunk))} tokens\n")
    return chunks

from transformers import AutoTokenizer

def split_into_chunks_tokenwise(text: str, chunk_token_size: int, model_name: str) -> list:
    """
    Split text into chunks by actual token count.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")    
    input_tokens = tokenizer.encode(text)

    chunks = []
    for i in range(0, len(input_tokens), chunk_token_size):
        chunk_tokens = input_tokens[i: i + chunk_token_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

    print(f"Token-based split: {len(chunks)} chunks of max {chunk_token_size} tokens.")
    return chunks


def count_tokens(text: str, model: str = "llama-3.3-70b-versatile") -> int:
    """
    Count the number of words in a text string.
    
    Args:
        text: The input text
        model: Not used, kept for compatibility
        
    Returns:
        int: Number of words
    """
    return len(text.split())

# def get_default_prompts() -> tuple[str, str]:
    """
    Get default system prompts for worker and manager agents for multiple choice QA.

    Returns:
        tuple[str, str]: (worker_prompt, manager_prompt)
    """
    worker_prompt = """You are a worker agent. Your task is to analyze a section of the provided document and extract information relevant to the following multiple choice question. 
Read the question and all answer options carefully. 
If the answer can be found in your section, provide supporting evidence and indicate which option (1, 2, 3, or 4) is most supported by your section. 
If the answer is not present, state that clearly."""

    manager_prompt = """You are a manager agent. Your task is to review the analyses and evidence provided by the worker agents for a multiple choice question. 
Synthesize their findings and select the single best answer (1, 2, 3, or 4) to the question, providing a brief justification based on the workers' evidence. 
If the answer is uncertain, choose the most likely option and explain your reasoning."""

    return worker_prompt, manager_prompt

def get_default_prompts() -> tuple[str, str]:

    """
    Get default system prompts for worker and manager agents.
    
    Returns:
        tuple[str, str]: (worker_prompt, manager_prompt)
    """
    prompt = """You are a code debugger"""

    return prompt