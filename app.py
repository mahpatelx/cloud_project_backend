from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from huggingface_hub import login

# Initialize Flask app
app = Flask(__name__)

# Model configuration
model_name = "meta-llama/Llama-3.2-1B"

# Login to Hugging Face Hub
login(token="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")

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

# Create a pipeline for text generation
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # Ensure this matches your CUDA device
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
        template = PromptTemplate(input_variables=["prompt"], template="{prompt}")

        # Create an LLMChain with the prompt template and LLM
        llm_chain = LLMChain(prompt=template, llm=llm)

        # Generate text using the chain
        generated_text = llm_chain.run(prompt=prompt)
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
