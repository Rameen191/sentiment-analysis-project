import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text, verbose=False):
    step1 = text.lower()
    if verbose: print(f"1. Lowercase:\n{step1[:100]}\n")

    step2 = re.sub(r'[^\w\s]', '', step1)
    if verbose: print(f"2. Remove Punctuation:\n{step2[:100]}\n")

    step3 = re.sub(r'\d+', '', step2)
    if verbose: print(f"3. Remove Numbers:\n{step3[:100]}\n")

    step4 = word_tokenize(step3)
    if verbose: print(f"4. Tokenized:\n{step4[:15]}\n")

    step5 = [w for w in step4 if w not in stop_words]
    if verbose: print(f"5. Stopwords Removed:\n{step5[:15]}\n")

    step6 = [lemmatizer.lemmatize(w) for w in step5]
    if verbose: print(f"6. Lemmatized:\n{step6[:15]}\n")

    return ' '.join(step6)