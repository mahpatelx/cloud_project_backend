from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFacePipeline
from huggingface_hub import login

# Model configuration

# model_name = "meta-llama/Llama-3.2-1B"
model_name = "meta-llama/Llama-3.2-1B-Instruct"

with open("env.txt", "r") as f:
    token = f.read().strip()  # Strip any extra whitespace or line breaks
    print(token)  # Optional: Check the token
# Login to Hugging Face Hub
login(token=token)




# Initialize Flask app
app = Flask(__name__)


# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a pipeline for text generation (CPU-friendly)
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # Use -1 to run on CPU
)

# Wrap the pipeline with LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=text_pipeline)

@app.route('/chat', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Define a prompt template
        template = PromptTemplate(input_variables=["prompt"], template="You are helpful assistant. {prompt}")

        # Create an LLMChain with the prompt template and LLM
        llm_chain = LLMChain(prompt=template, llm=llm)

        # Generate text using the chain
        generated_text = llm_chain.run(prompt=prompt)
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

