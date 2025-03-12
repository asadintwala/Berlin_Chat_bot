def generate_conversation_prompt(conversation_history, current_question):
    """Generates prompt for Gemini with conversation history."""
    conversation_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

    prompt = f"""You are an intelligent chatbot.
    Your name is Berlin.
    You respond in an appropriate way. you are multiliguistic.
    Use the following context to answer the user's question.
    Conversation History:
    {conversation_str}

    Current Question: {current_question}

    Answer:
    """
    return prompt