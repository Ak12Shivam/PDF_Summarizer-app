import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import tempfile
import os
import torch

# Configure Streamlit page
st.set_page_config(
    page_title="PDF Summarizer & QA",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'qa_context' not in st.session_state:
    st.session_state.qa_context = None

@st.cache_resource
def load_models():
    """Load and cache both BART and BERT models"""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        qa_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)
        return summarizer, qa_pipeline
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name

        pdf_document = fitz.open(tmp_file_path)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        
        pdf_document.close()
        os.unlink(tmp_file_path)
        
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def summarize_text(summarizer, text, max_length, min_length=None):
    """Summarize text using BART"""
    if min_length is None:
        min_length = max(30, max_length // 2)
    
    try:
        # Split text into chunks if it's too long
        max_chunk_length = 1024  # BART model's maximum input length
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        summaries = []
        for chunk in chunks:
            if len(chunk.strip()) > 50:  # Only summarize chunks with substantial content
                summary = summarizer(chunk, 
                                  max_length=max_length, 
                                  min_length=min_length, 
                                  do_sample=False)
                summaries.append(summary[0]['summary_text'])
        
        return ' '.join(summaries)
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return None

def answer_question(qa_pipeline, context, question):
    """Answer questions about the text using BERT"""
    try:
        # Split context into chunks if it's too long
        max_length = 512  # BERT's maximum sequence length
        words = context.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1  # +1 for space
            if current_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Get answer from each chunk and return the one with highest confidence
        best_answer = None
        best_score = 0
        
        for chunk in chunks:
            result = qa_pipeline(question=question, context=chunk)
            if result['score'] > best_score:
                best_score = result['score']
                best_answer = result
        
        return best_answer
    except Exception as e:
        st.error(f"Error during question answering: {str(e)}")
        return None

def main():
    st.title("📄 PDF Summarizer & QA Assistant")
    st.write("Upload a PDF file to get an AI-generated summary and ask questions about its contents.")
    
    # Initialize models
    summarizer, qa_pipeline = load_models()
    if summarizer is None or qa_pipeline is None:
        st.error("Failed to load the AI models. Please try again later.")
        return

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            if st.session_state.extracted_text is None:
                st.session_state.extracted_text = extract_text_from_pdf(uploaded_file)
            
            if st.session_state.extracted_text:
                # Show PDF details
                st.success("PDF successfully processed!")
                word_count = len(st.session_state.extracted_text.split())
                st.info(f"Document contains approximately {word_count} words")
                
                # Create tabs for different functionalities
                tab1, tab2 = st.tabs(["Summarize", "Ask Questions"])
                
                # Summarization tab
                with tab1:
                    with st.expander("Show extracted text"):
                        st.text_area("Extracted Text", st.session_state.extracted_text, height=200)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        max_words = st.slider(
                            "Summary length (words)", 
                            min_value=50, 
                            max_value=500, 
                            value=150, 
                            step=50
                        )
                    
                    if st.button("Generate Summary", type="primary"):
                        with st.spinner("Generating summary..."):
                            summary = summarize_text(
                                summarizer,
                                st.session_state.extracted_text,
                                max_length=max_words
                            )
                            
                            if summary:
                                st.write("### Summary")
                                st.write(summary)
                                st.code(summary, language=None)
                                
                                original_length = len(st.session_state.extracted_text.split())
                                summary_length = len(summary.split())
                                reduction = ((original_length - summary_length) / original_length) * 100
                                st.success(f"Reduced document length by {reduction:.1f}%")
                
                # Question Answering tab
                with tab2:
                    st.write("### Ask Questions About the Document")
                    question = st.text_input("Enter your question about the document:")
                    
                    if question:
                        if st.button("Get Answer", type="primary"):
                            with st.spinner("Finding answer..."):
                                answer = answer_question(qa_pipeline, st.session_state.extracted_text, question)
                                
                                if answer:
                                    st.write("### Answer")
                                    st.write(answer['answer'])
                                    st.progress(answer['score'])
                                    st.caption(f"Confidence score: {answer['score']:.2%}")
                                    
                                    # Show context
                                    with st.expander("Show answer context"):
                                        start_idx = max(0, answer['start'] - 100)
                                        end_idx = min(len(st.session_state.extracted_text), answer['end'] + 100)
                                        context = st.session_state.extracted_text[start_idx:end_idx]
                                        st.write(context)

if __name__ == "__main__":
    main()