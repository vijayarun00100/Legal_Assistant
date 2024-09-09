import PyPDF2
def count_words_in_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        word_count = 0
        for page in reader.pages:
            text = page.extract_text()
            words = text.split()
            word_count += len(words)
    return word_count
pdf_path = 'D:\law\commercial courts rules , act.pdf'
word_count = count_words_in_pdf(pdf_path)
print(f"Total number of words in the PDF: {word_count}") 


