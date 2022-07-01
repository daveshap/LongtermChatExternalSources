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


def save_debug(label, content):
    filename = '%s_%s.txt' % (time(), label)
    with open('debug/%s' % filename, 'w') as outfile:
        outfile.write(content)


def search_index(recent, all_lines, count=10):
    if len(all_lines) <= count:
        return list()
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
        return [i['line'] for i in ordered]
    except:
        return [i['line'] for i in ordered]


def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'TIM:']):
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
    if len(answers) == 1:
        return answers[0]
    answer = ' '.join(answers)
    prompt = open_file('prompt_merge.txt').replace('<<QUESTION>>', question).replace('<<ANSWERS>>', answer)
    answer = gpt3_completion(prompt)
    return answer


if __name__ == '__main__':
    conversation = list()
    while True:
        # get user input and vector
        user_says = input("USER: ")
        line_in = 'USER: %s' % user_says
        vector = gpt3_embedding(line_in)
        info = {'line': line_in, 'vector': vector}
        conversation.append(info)
        # search conversation for previous relevant lines of dialog
        old_lines = search_index(info, conversation, 10)
        recent_conversation = [i['line'] for i in conversation]
        if len(recent_conversation) > 30:
            recent_conversation = recent_conversation[-30:0]
        convo_block = '\n'.join(old_lines) + '\n' + '\n'.join(recent_conversation)
        convo_block = convo_block.strip()
        save_debug('convo', convo_block)
        # generate a search query to find external article in the wide world
        prompt = open_file('prompt_wikipedia.txt').replace('<<BLOCK>>', convo_block)
        title = gpt3_completion(prompt)
        save_debug('wiki title', title)
        wiki = fetch_wiki(title).content.encode(encoding='ASCII',errors='ignore').decode()
        save_debug('wiki article', wiki)
        # generate a specific follow-up question to use to query the external information
        prompt = open_file('prompt_followup.txt').replace('<<BLOCK>>', convo_block).replace('<<TOPIC>>', title)
        question = gpt3_completion(prompt)
        save_debug('question', question)
        answer = answer_question(wiki, question)
        save_debug('answer', answer)
        # populate the chat prompt
        prompt = open_file('prompt_chat.txt').replace('<<BLOCK>>', convo_block).replace('<<HINT>>', answer)
        response = gpt3_completion(prompt)
        # save the output
        vector = gpt3_embedding(response)
        line_out = 'TIM: %s' % response
        info = {'line': line_out, 'vector': vector}
        conversation.append(info)
        print(line_out)