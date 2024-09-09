import re
import nltk
import numpy as np
import pandas as pd
import time
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from transformers import pipeline
pipe = pipeline("fill-mask", model="nlpaueb/legal-bert-base-uncased")

# Load model directly
from transformers import AutoTokenizer, AutoModelForPreTraining
import pickle  
nltk.download('stopwords')
nltk.download('punkt')
import PyPDF2

#get text from the pdf using Pypdf 

string = ""
def words_in_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        word_count = 0
        for page in reader.pages:
            text = page.extract_text()
            words = text.split()
            string += words
    return string

pdf_path = 'D:\law\commercial courts rules , act.pdf'
words = words_in_pdf(pdf_path)

def preprocess_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^A-Za-z]+", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
    text = " ".join(tokens)
    text = text.lower().strip()
    return text


df = pd.DataFrame({
    'text': ["Sample sentence 1.", "Another example sentence!", "Third text goes here."],
    'target': [0, 1, 0]
})


df['text_cleaned'] = df['text'].apply(lambda text: preprocess_text(words))
df = df[df['text_cleaned'] != '']

# Load SentenceTransformer model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModelForPreTraining.from_pretrained("nlpaueb/legal-bert-base-uncased")


df['encode_transformers'] = df['text_cleaned'].apply(lambda text: model.encode(text, convert_to_numpy=True).flatten())

X_transformers = np.vstack(df['encode_transformers'])

truelabel = df['target'].values.tolist()


with open('word_embeddings.pkl', 'wb') as f:
    pickle.dump(X_transformers, f)
    
with open('labels.pkl', 'wb') as f:
    pickle.dump(truelabel, f)

st = time.time()
et = time.time()
print("Elapsed time: {:.2f} seconds".format(et - st))


# To load the saved embeddings later:
# with open('word_embeddings.pkl', 'rb') as f:
#     X_transformers = pickle.load(f)

