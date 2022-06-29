import os
import openai
import json
import numpy as np
import textwrap
import re
from time import time,sleep
import wikipedia


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = open_file('openaiapikey.txt')


def gpt3_embedding(content, engine='text-similarity-ada-001'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):  # return dot product of two vectors
    return np.dot(v1, v2)


def search_index(recent, all_lines, count=10):
    scores = list()
    for i in all_lines:
        if recent['vector'] == i['vector']:
            continue
        score = similarity(recent['vector'], i['vector'])
        #print(score)
        scores.append({'line': i['line'], 'score': score})
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered


def chat_completion(prompt):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(model="davinci:ft-david-shapiro:tutor-2022-05-16-22-02-24",
                                                prompt=prompt,
                                                temperature=0.7,
                                                max_tokens=100,
                                                stop=["USER:", "TIM:"])
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(5)


def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=200, freq_pen=0.0, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def fetch_wiki(title):
    try:
        page = wikipedia.page(title)
        return page
    except:
        search = wikipedia.search(title)
        page = wikipedia.page(search[0])
        return page


def answer_question(article, question):
    chunks = textwrap.wrap(article, 10000)
    answers = list()
    for chunk in chunks:
        prompt = open_file('prompt_answer.txt').replace('<<PASSAGE>>', chunk).replace('<<QUESTION>>', question)
        answer = gpt3_completion(prompt)
        answers.append(answer)
    answer = ' '.join(answers)
    prompt = open_file('prompt_merge.txt').replace('<<QUESTION>>', question).replace('<<ANSWERS>>', answer)
    answer = gpt3_completion(prompt)
    return answer


if __name__ == '__main__':
    conversation = list()
    while True:
        # get user input and vector
        user_says = input("USER:")
        line_in = 'USER: %s' % user_says
        vector = gpt3_embedding(line_in)
        info = {'line': line_in, 'vector': vector}
        conversation.append(info)
        # search conversation for previous relevant lines of dialog
        old_lines = search_index(info, conversation, 10)
        if len(conversation) > 30:
            recent_conversation = conversation[-30:0]
        convo_block = '\n'.join(old_lines) + '\n' + '\n'.join(recent_conversation)
        # generate a search query to find external article in the wide world
        prompt = open_file('prompt_wikipedia.txt').replace('<<BLOCK>>', convo_block)
        title = gpt3_completion(prompt)
        wiki = fetch_wiki(title).content
        # generate a specific follow-up question to use to query the external information
        prompt = open_file('prompt_followup.txt').replace('<<BLOCK>>', convo_block).replace('<<TOPIC>>', title)
        question = gpt3_completion(prompt)
        answer = answer_question(wiki, question)
        