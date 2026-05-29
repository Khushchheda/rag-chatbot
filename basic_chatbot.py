from transformers import pipeline

chatbot = pipeline("text-generation", model="distilgpt2")

print("AI Chatbot Started (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    prompt = f"You are a rude AI assistant.\nUser: {user_input}\nAI:"

    response = chatbot(
        prompt,
        max_length=100,
        num_return_sequences=1
    )

    print("\nAI:", response[0]["generated_text"], "\n")