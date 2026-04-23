import subprocess
import sys
from openai import OpenAI

MODEL_NAME = "qwen3-vl:4b"

def main():
    print(f"Pulling model {MODEL_NAME} via Ollama...")
    try:
        # Run ollama pull
        subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
        print(f"\nModel {MODEL_NAME} pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nFailed to pull model: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("\nOllama executable not found. Please ensure Ollama is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)

    print("\nTesting model connection via OpenAI compatible endpoint...")
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello, are you running?"}],
            max_tokens=500
        )
        print("\nResponse from model:")
        message = completion.choices[0].message
        if hasattr(message, 'reasoning') and message.reasoning:
            print("[Reasoning]:")
            print(message.reasoning)
            print("\n[Content]:")
        print(message.content)
        print("\nSetup and test complete! You can now run the inference script.")
    except Exception as e:
        print(f"\nFailed to communicate with model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
