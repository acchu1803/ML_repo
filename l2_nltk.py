from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.chunk import RegexpParser

s = '''Good muffins cost $3.88 \n in New York.  Please buy me ... two of them.\n \n Thanks.'''

word_tokenize(s) 
print(word_tokenize(s) )

nltk.download('stopwords')
print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(s)
filtered_sentence = []
 
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)
 
ps = PorterStemmer()
words = ["program", "programs", "programmer", "programming", "programmers"]
 
for w in words:
    print(w, " : ", ps.stem(w))

s = '''Good muffins cost $3.88 \n in New York.  Please buy me ... two of them.\n \n Thanks.'''

word_tokens = word_tokenize(s) 

tagged = nltk.pos_tag(word_tokens)
print(tagged)

chunk_patterns = r"""
    NP: {<DT>?<JJ>*<NN>}  
    VP: {<VB.*><NP|PP>} 
"""
nltk.download('averaged_perceptron_tagger_eng')

namedEnt = nltk.ne_chunk(tagged, binary=False)
namedEnt.draw()