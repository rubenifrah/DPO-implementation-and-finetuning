
import sys

def extract_text(pdf_path):
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        pass

    try:
        import PyPDF2
        reader = PyPDF2.PdfFileReader(pdf_path)
        text = ""
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText() + "\n"
        return text
    except ImportError:
        pass

    return "Error: Could not import pypdf or PyPDF2"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    print(extract_text(pdf_path))
