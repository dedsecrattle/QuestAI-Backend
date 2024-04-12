from flask import Flask, request, jsonify
import os
import vertexai
from flask_cors import CORS
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from vertexai.language_models import TextGenerationModel

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/etc/secrets/googleauth.json"

app = Flask(__name__)
cors = CORS(app)

vertexai.init(project="cs3263-project", location="us-central1")
model = GenerativeModel("gemini-1.0-pro-002")
chat = model.start_chat()


@app.route('/')
def Home():
    return "Welcome to QuestAI backend"
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    topic = data.get('topic', '')
    subtopic = data.get('subtopic', '')
    level = data.get('level', '')
    max_output_tokens = data.get('max_output_tokens', 3729)
    temperature = data.get('temperature', 0.9)
    top_p = data.get('top_p', 1)

    harm_categories = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }



    prompt = f"You're a Helpful AI Assistant helping students to learn about new topics , Generate Learning content on the topic {topic} and subtopic {subtopic} in around 1500 words for a {level} level student."
    generation_config = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    response = model.generate_content(
        [prompt],
        generation_config=generation_config,
        safety_settings=harm_categories,
        stream=False,
    )

    return jsonify({"generated_content": response.text})


@app.route('/quiz', methods=['POST'])
def quiz():
    data = request.get_json()
    topic = data.get('topic', '')
    subtopic = data.get('subtopic', '')
    level =  data.get('level', '')
    max_output_tokens = data.get('max_output_tokens', 3729)
    temperature = data.get('temperature', 0.9)
    top_p = data.get('top_p', 1)

    harm_categories = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    prompt = f"Generate 10 MCQ Question on topic {topic} and subtopic {subtopic} with options and correct option for a {level} level Student"
    generation_config = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    response = model.generate_content(
        [prompt],
        generation_config=generation_config,
        safety_settings=harm_categories,
        stream=False,
    )

    return jsonify({"generated_content": response.text})


@app.route('/summary', methods=['POST'])
def summary():
    data = request.get_json()
    text = data.get('extracted_text', '')
    max_output_tokens = data.get('max_output_tokens', 3729)
    temperature = data.get('temperature', 0.9)
    top_p = data.get('top_p', 1)

    prompt = f"Summarise the following text in around 50% of the original length- {text}"
    parameters = {
    "max_output_tokens": 2000,
    "temperature": 0.1,
    "top_p": 1
    }

    model = TextGenerationModel.from_pretrained("text-bison-32k")
    response = model.predict(
        prompt,**parameters
    )
    return jsonify({"generated_content": response.text})


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    context = data.get('context', '')
    question = data.get('question', '')
    max_output_tokens = data.get('max_output_tokens', 3729)
    temperature = data.get('temperature', 0.9)
    top_p = data.get('top_p', 1)

    harm_categories = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    generation_config = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    prompt = f"{context}  - Given the above text answer the following question: {question}"

    response = model.generate_content(
        [prompt],
        generation_config=generation_config,
        stream=False,
        safety_settings=harm_categories
    )
    return jsonify({"generated_content": response.text})

if __name__ == '__main__':
    app.run(debug=True)


