import PyPDF2

class DataIngestion:

    def __init__(self):
        pass

    def text_extractor(self,file):

        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text