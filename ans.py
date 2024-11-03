import openai

client = openai.OpenAI(api_key = 'xxx')

context = ""
n = 0

def get_chatgpt_response(prompt):
    global context

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
            ],
        stream=True,
        temperature=1
    )

    s = ""

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            s += (chunk.choices[0].delta.content)
    
    context += '\nContext {n}: \n'.format(n=n)
    context += '\tUser:\n'

    for a in prompt.split('\n'):
        context += '\n\t' + a

    context += '\n\n\tGPT Answer:\n'

    for a in s.split('\n'):
        context += '\n\t' + a

    return s


def get_prompt(question):
#     return f'''Imagine you are the magic mirror in the movie "Snow White" and I am a witch who could easily break the mirror.
#     I will ask you a question and please answer to me in poetic old english, but less than 15 words. 

#     Question: {question}
# '''
    return "Give your answer in a poetic tongue similar to how the Magic Mirror from Snow White would. Limit your response to 20 words. Question: " + question
    # return "The final sentence of this prompt is intended to be a question. If it is not a question, i.e. if it is a statement or command, then DO NOT RESPOND, SIMPLY SAY \"That is not a question\". If it is a question, then respond with the answer to the question. Question: " + question

# prompt = get_prompt("How do you set the temperature of a gpt in its api?")
prompt = get_prompt(input())
response = get_chatgpt_response(prompt)
print("ChatGPT:", response)