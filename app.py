from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from googletrans import Translator

app = Flask(__name__)

from itertools import permutations

def jaro_winkler(s1: str, s2: str, p: float = 0.1):
    def jaro(s1, s2):
        len_s1, len_s2 = len(s1), len(s2)
        if len_s1 == 0 and len_s2 == 0:
            return 1.0
        if len_s1 == 0 or len_s2 == 0:
            return 0.0
        
        match_distance = max(len_s1, len_s2) // 2 - 1
        s1_matches = [False] * len_s1
        s2_matches = [False] * len_s2
        matches = 0
        
        for i in range(len_s1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len_s2)
            for j in range(start, end):
                if s2_matches[j]:
                    continue
                if s1[i] == s2[j]:
                    s1_matches[i] = True
                    s2_matches[j] = True
                    matches += 1
                    break
        
        if matches == 0:
            return 0.0
        
        k = 0
        transpositions = 0
        for i in range(len_s1):
            if s1_matches[i]:
                while not s2_matches[k]:
                    k += 1
                if s1[i] != s2[k]:
                    transpositions += 1
                k += 1
        
        transpositions //= 2
        
        return ((matches / len_s1) + (matches / len_s2) + ((matches - transpositions) / matches)) / 3.0
    
    jaro_dist = jaro(s1, s2)
    prefix_len = 0
    max_prefix_len = min(4, min(len(s1), len(s2)))
    
    for i in range(max_prefix_len):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    
    return jaro_dist + (prefix_len * p * (1 - jaro_dist))

def jaro_winkler_permutations(s1: str, s2: str, p: float = 0.1):
    words1 = s1.lower().split()
    words2 = s2.lower().split()
    
    max_similarity = 0.0
    for perm1 in permutations(words1):
        for perm2 in permutations(words2):
            perm1_str = ' '.join(perm1)
            perm2_str = ' '.join(perm2)
            similarity = jaro_winkler(perm1_str, perm2_str, p)
            max_similarity = max(max_similarity, similarity)
    return max_similarity

def jaccard_similarity(str1, str2, n=1):
    # Function to generate n-grams from a string
    def generate_ngrams(string, n):
        ngrams = set()
        for i in range(len(string) - n + 1):
            ngrams.add(string[i:i + n])
        return ngrams
    
    # Tokenize strings into sets of n-grams
    set1 = generate_ngrams(str1, n)
    set2 = generate_ngrams(str2, n)
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# def detect_language(text):
    
#     detection = translator.detect(text)
    
#     print("Text:", text)
#     print("Detected language:", detection.lang)
#     print("Confidence:", detection.confidence)

df_conv_full = pd.read_csv('data/conv_full.csv')
translator = Translator()

# Example messages list
messages = [
    {"sender": "person_2", "content": "Hi there! I'm Jemmy, I would love to talk with you :D"}
]

@app.route('/')
def index():
    return render_template('chat.html', messages=messages)

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    sender = data.get('sender')
    content = data.get('content')
    if sender and content:
        message = {'sender': sender, 'content': content}
        messages.append(message)

        content = content.lower()

        origin_lang = translator.detect(content).lang

        # Only English and Indonesia
        if origin_lang != 'en' and origin_lang != 'id':
            response_text = "sorry, i only understand english and indonesia"
        else:
            if origin_lang != 'en':
                content = translator.translate(content, dest='en').text

            print(translator.translate(content, dest='en').text)

            df_response = df_conv_full.copy()
            list_response = [jaccard_similarity(df_response.iloc[i, 0], content) for i in range(len(df_response))]
            df_response['similarity'] = list_response
            df_response = df_response.sort_values(by='similarity', ascending=False).reset_index(drop=True)
            df_response = df_response[:3000]
            df_response['jaro_winkler_similarity'] = [jaro_winkler(df_response.iloc[i, 0], content, 0.1) for i in range(len(df_response))]
            df_response = df_response.sort_values(by='jaro_winkler_similarity', ascending=False).reset_index(drop=True)

            best_similarity = df_response['jaro_winkler_similarity'].max()
            df_response = df_response[df_response['jaro_winkler_similarity'] == best_similarity]

            print(f"Response: {np.random.choice(df_response['answer'].values)}")

            response_text = np.random.choice(df_response['answer'].values)
        
            response_text = translator.translate(response_text, dest=origin_lang).text.lower()

        response = {'sender': 'person_2', 'content': response_text}

        messages.append(response)
        
        return jsonify({'status': 'success', 'message': message, 'response': response})
    return jsonify({'status': 'error'}), 400

if __name__ == '__main__':
    app.run(debug=True)
