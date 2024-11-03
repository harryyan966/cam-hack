import openai

API_KEY='xxx'


class MirrorBrain():
    def __init__(self, api_key):
        self.gpt = openai.OpenAI(api_key = api_key)
        self.message_count = 0
        self.conversation = ''
    
    def get_prompt(self, question):
        return f'''Give your answer in a poetic tongue similar to how the Magic Mirror from Snow White would. Limit your response to 20 words. Question: "{question}"'''
    
    def respond(self, question):
        prompt = self.get_prompt(question)

        stream = self.gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.conversation},
                {"role": "user", "content": prompt},
                ],
            stream=True,
            temperature=1
        )

        s = ""

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                s += (chunk.choices[0].delta.content)
        
        self.conversation += '\nContext {n}: \n'.format(n=self.message_count)
        self.conversation += '\tUser:\n'

        for a in prompt.split('\n'):
            self.conversation += '\n\t' + a

        self.conversation += '\n\n\tGPT Answer:\n'

        for a in s.split('\n'):
            self.conversation += '\n\t' + a
        
        self.message_count += 1

        return s
        
