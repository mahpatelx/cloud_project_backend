from huggingface_hub import login
login(token="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")



# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch



# # Load the model and tokenizer
# model_name = "llama-3.2-1b"  # Replace with the actual model name if different
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# def generate_text(prompt, max_length=5000, temperature=0.3, num_return_sequences=1, do_sample=True):
#     # Encode the input prompt
#     inputs = tokenizer(prompt, return_tensors="pt")

#     # Generate text
#     outputs = model.generate(
#         inputs.input_ids,
#         max_length=max_length,
#         temperature=temperature,
#         num_return_sequences=num_return_sequences,
#         do_sample=do_sample
#     )

#     # Decode the generated text
#     generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return generated_texts

# # Example usage
# if __name__ == "__main__":

#     prompt = "Once upon a time"
#     max_length = 5000
#     temperature = 0.3
#     num_return_sequences = 1
#     do_sample = True

#     generated_texts = generate_text(prompt, max_length, top_k, top_p, temperature, num_return_sequences, do_sample)

#     for i, text in enumerate(generated_texts):
#         print(f"Generated Text {i+1}:\n{text}\n")
