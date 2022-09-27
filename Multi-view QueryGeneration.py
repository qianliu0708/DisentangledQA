import json
def get_file_contents(filename, encoding='utf-8'):
    filename=filename.encode('utf-8')
    with open(filename, encoding=encoding) as f:
        content = f.read()
    return content
def read_json_file(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)
def write_json_to_file(json_object, json_file, mode='w', encoding='utf-8'):
    with open(json_file, mode, encoding=encoding) as outfile:
        json.dump(json_object, outfile, indent=4, sort_keys=True, ensure_ascii=False)
from transformers import pipeline
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
STOPWORDS = stopwords.words("english")
STOPWORDS.append("would")
STOPWORDS.append("could")
STOPWORDS = [stopword + " " for stopword in STOPWORDS]
def clean_query(query, remove_stopwords=True,remove_qmark = True):
    if remove_qmark:
        if query[-1] == "?":
            query = query[0:-1]
    if remove_stopwords:
        query_split = query.split()
        new_query_split = []
        for word in query_split:
            if word.lower() + " " not in STOPWORDS:
                new_query_split.append(word)
        query = " ".join(new_query_split)
    return query
def Gen_Ngram(sentence,n=4,m=1):
    if len(sentence)<n:
        n = len(sentence)
    ngrams = [sentence[i-k:i] for k in range(m, n+1) for i in range(k, len(sentence)+1)]
    return ngrams
if __name__ == '__main__':
    nlp = pipeline("ner",grouped_entities=True)

    for file in ['dev','train','test']:
        count=0
        orig_data = read_json_file("data/{}.json".format(file))
        query_dict = {}
        #step1:NER
        for item in tqdm(orig_data):
            new_item = {}
            qid = item['qid']
            question = item['question']
            question_cleaned = clean_query(question)
            new_item['question'] = question
            new_item['clean_q'] = question_cleaned
            #step1:NER
            ner_list = nlp(question_cleaned)
            new_item['ner_query'] = [per_ner['word'] for per_ner in ner_list]
            #step2:Ngram--hard match titles
            ngram_list = Gen_Ngram(question.replace("?","").split(),m=1,n =len(question.split()))
            ngram_str = [" ".join(ngram) for ngram in ngram_list]
            Final_ngram = []
            for per_str in ngram_str:
                clean = clean_query(per_str)
                if len(clean)!=0:
                    Final_ngram.append(clean)
            new_item['ngram_query'] = Final_ngram
            #step3:Noun
            nouns = []
            for word, pos in nltk.pos_tag(nltk.word_tokenize(question.replace("?",""))):
                if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                    nouns.append(word)
            new_item['noun_query'] = " ".join(nouns)
            query_dict[qid] = new_item
        write_json_to_file(query_dict,"data/ir_info/query_{}.json".format(file))