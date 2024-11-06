import cohere
import openai
import random

# Set your API keys here
cohere_api_key = "COHERE-KEY"
openai_api_key = "OPENAI-KEY"
co = cohere.Client(cohere_api_key)
openai.api_key = openai_api_key

def generate_mixed_accuracy_response(user_prompt):
    """
    Generates a response with a 25% chance of being fully accurate using Cohere.
    Otherwise, the response includes subtle inaccuracies.
    """
    # 25% chance for an accurate response
    if random.random() < 0.25:
        accuracy_style = "Provide an accurate and factual response to the following prompt:"
    else:
        accuracy_style = (
            "Simulate a response that subtly mixes true and false information, "
            "making it difficult to immediately verify its accuracy. Provide an answer to the following prompt with slight inaccuracies:"
        )
    
    generation_prompt = f"{accuracy_style} {user_prompt}"
    
    response = co.generate(
        model="command-xlarge-nightly",  # Use an appropriate model name from Cohere
        prompt=generation_prompt,
        max_tokens=150,
        temperature=0.7
    )
    
    return response.generations[0].text.strip()

def verify_response_for_hallucination(response):
    """
    Uses OpenAI's gpt-3.5-turbo or gpt-4-turbo model to check for hallucinations/confabulations.
    Starts with a veracity score, then lists only the incorrect portions with justifications.
    """
    verification_prompt = (
        "Evaluate the following statement for hallucinations and confabulations. "
        "Start by assigning a total veracity/confabulation score as a percentage, reflecting the overall accuracy. "
        "Then, list only the factually incorrect portions. For each incorrect statement, include a concise explanation of why it is incorrect.\n\n"
        f"Statement: {response}"
    )
    
    # Using OpenAI's Chat Completion API with gpt-3.5-turbo
    verification = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4-turbo" if available
        messages=[
            {"role": "system", "content": "You are an AI that evaluates text for factual accuracy."},
            {"role": "user", "content": verification_prompt}
        ],
        max_tokens=500,
        temperature=0.5
    )
    
    return verification.choices[0].message['content'].strip()

def hallucination_check(user_prompt):
    """
    Generates a potentially mixed-accuracy response based on the user prompt using Cohere
    and verifies it for hallucinations/confabulations using OpenAI, displaying only incorrect parts with a veracity score.
    """
    # Step 1: Generate a potentially hallucinated response with Cohere
    generated_response = generate_mixed_accuracy_response(user_prompt)
    print("Generated Response:")
    print(generated_response)

    # Step 2: Verify the generated response for accuracy with OpenAI
    verification_result = verify_response_for_hallucination(generated_response)
    
    # Cleaned Output Formatting
    print("\nVerification Result:")
    print("===================================================")
    print("Veracity Score:")
    print("---------------------------------------------------")
    print(verification_result.split('\n')[0])  # Display the score as the first line
    
    print("\nIncorrect Statements:")
    print("---------------------------------------------------")
    # Display only the hallucinated parts, skipping the score
    for line in verification_result.split('\n')[1:]:
        print(line.strip())
    print("===================================================")

# Example usage
if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    hallucination_check(user_prompt)