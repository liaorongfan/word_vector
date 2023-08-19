import pdfplumber


def extract_pdf(pdf_path):
    """
    Extract text from pdf
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        num_pages = len(pdf.pages)
        page_contents = []
        for page in pdf.pages:
            page_contents.append(page.extract_text())
    txt_path = pdf_path.replace(".pdf", ".txt")
    with open(txt_path, "w") as f:
        for page in page_contents:
            f.write(page + "\n")


if __name__ == '__main__':
    import glob
    pdfs = glob.glob("data/pdf/*/*.pdf")
    for pdf in pdfs:
        extract_pdf(pdf)


