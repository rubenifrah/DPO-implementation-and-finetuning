import sys

def extract_text_from_pdf(pdf_path):
    """
    Attempts to extract text from a PDF file.
    
    Robustness:
    It tries to use 'pypdf' first (the modern library).
    If that fails, it falls back to 'PyPDF2' (older legacy library).
    """
    
    # Method 1: Try pypdf (Preferred)
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return "\n".join(text)
    except ImportError:
        pass # Fallthrough to next method
    except Exception as e:
        print(f"pypdf failed: {e}")

    # Method 2: Try PyPDF2 (Legacy)
    try:
        import PyPDF2
        reader = PyPDF2.PdfFileReader(pdf_path)
        text = []
        for page_num in range(reader.numPages):
            text.append(reader.getPage(page_num).extractText())
        return "\n".join(text)
    except ImportError:
        pass
    except Exception as e:
        print(f"PyPDF2 failed: {e}")

    return "Error: Could not import pypdf or PyPDF2, or extraction failed."

if __name__ == "__main__":
    # Simple CLI usage
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf.py <path_to_pdf>")
        sys.exit(1)
    
    target_pdf = sys.argv[1]
    extracted_text = extract_text_from_pdf(target_pdf)
    
    print("-" * 40)
    print(f"Extracted content from: {target_pdf}")
    print("-" * 40)
    print(extracted_text)
