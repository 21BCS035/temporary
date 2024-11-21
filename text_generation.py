from huggingface_hub import InferenceClient

client = InferenceClient(api_key="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")



def generate_text(prompt):
    messages = [
	{
		"role": "user",
		"content": prompt
	}
    ]

    stream = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
        messages=messages, 
        max_tokens=500,
        stream=True
    )
    result = ''
    for chunk in stream:
        result += chunk.choices[0].delta.content
    return result