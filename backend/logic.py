# backend/logic.py

from backend.emotion import capture_emotion
from backend.llama_engine import generate_response

def chat_session():
    """
    Runs the chat loop: prompts user input, captures emotion each turn,
    and prints an emotion-aware response until the user exits.
    """
    print("🤖 Welcome to your Emotion-Aware Mental Health Chatbot!")
    print("Type 'exit' or 'quit' to end the chat.\n")

    while True:
        # 1. Get user input
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        # 2. Capture emotion after user response
        emotion = capture_emotion()
        if not emotion:
            print("⚠️ No emotion detected, defaulting to 'neutral'.")
            emotion = "neutral"
        print(f"🧠 Detected emotion: {emotion}")

        # 3. Generate and print AI response
        reply = generate_response(user_input, emotion)
        print(f"Bot: {reply}\n")
