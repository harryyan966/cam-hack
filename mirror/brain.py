import openai



class MirrorBrain():
    def __init__(self, api_key):
        self.gpt = openai.OpenAI(api_key = api_key)
        self.message_count = 0
        self.conversation = ''
    
    def get_prompt(self, question):
        # return f"Give your answer to this question in at most 15 words. Question: {question}"
        return f'''Give your answer in a poetic tongue similar to what the Magic Mirror did in "Snow White". limit your response to at most 5 words. Question: "{question}"'''
    
    def respond(self, question):
        if 'hackathon' in question or 'win' in question:
            return 'Absolutely not you.'
        
        if 'ugliest' in question:
            return 'You, of course'
    
        if 'fairest' in question:
            return 'Lips red as the rose. Hair black as ebony, Skin white as snow'
        
        if 'who are you' in question:
            return '''I'm Chat G P T, your virtual assistant! How can I help you today?'''

        prompt = self.get_prompt(question)

        stream = self.gpt.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.conversation},
                {"role": "user", "content": prompt},
                ],
            stream=True,
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

        print(f"🤣Mirror Answered: {s}")

        return s
        
