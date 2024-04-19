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
    temperature = data.get('temperature', 0.8)
    top_p = data.get('top_p', 0.7)

    harm_categories = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    prompt1 = f"You're a highly capable and knowledgeable AI assistant. Your task is to generate a comprehensive, engaging, and age-appropriate learning content on the topic of {topic} and subtopic {subtopic}."

    prompt2 = f"The content should be approximately 1500 words long and tailored for a {level} level student. Cover the key concepts, relevant examples, and practical applications in a clear, structured, and compelling manner. Ensure the content is educational, informative, and accessible to the target audience."

    prompt3 = "Provide a strong concluding section that summarizes the main points and leaves the reader with a clear understanding of the topic and include URL links to the citations for the sources used as well additional resources for furthe exploration."

   
    prompt = "\n\n".join([prompt1, prompt2, prompt3])

    generation_config = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p
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
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.3)

    harm_categories = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    prompt = (
        f"Generate 10 MCQ questions on the topic of {topic} and subtopic {subtopic}, with 4 options each and the correct option clearly marked. Structure the questions and answers as follows:\n\n"
        "- Provide 3 Easy, 4 Medium, and 3 Hard questions.\n"
        "- Each question should begin with <question> and end with </question>.\n"
        "- For each question, provide 4 options, each contained between the labels <option> and </option>.\n"
        "- Include an explanation for each question, begin with label <explanation> and end with label </explanation>.\n"
        "- At the end, list the correct answers in the format <answers>A,B,C,D,...</answers>, where each letter corresponds to the correct option for the respective question.\n\n"
        "Format Example 1:\n"
        "<question>What is the capital of France?</question>\n"
        "<option>A. Paris</option>\n"
        "<option>B. London</option>\n"
        "<option>C. Berlin</option>\n"
        "<option>D. Madrid</option>\n"
        "<explanation>Paris is the capital city of France.</explanation>\n\n"
        "<answers>A</answers>\n\n"
        "Format Example 2:\n"
        "<question>What is the largest planet in our solar system?</question>\n"
        "<option>A. Earth</option>\n"
        "<option>B. Mars</option>\n"
        "<option>C. Jupiter</option>\n"
        "<option>D. Saturn</option>\n"
        "<explanation>Jupiter is the largest planet in our solar system.</explanation>\n\n"
        "<answers>C</answers>"
        "The format above needs to be strictly followed, and you must include the closing </question> tag for each question."
    )
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

    prompt = f"You are a highly skilled summarizer tasked with creating informative 1000-word summary of the following text: {text}. Ensure the summary captures the key ideas, main arguments, and essential information in a clear and structured manner. Highlight the most important points, while omitting unnecessary details. The summary should be accessible and valuable for the reader, providing a comprehensive overview of the source material."
    parameters = {
    "max_output_tokens": 2500,
    "temperature": 0.7,
    "top_p": 0.9,
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
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.8)

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

    prompt = f"You are an AI assistant with extensive knowledge on a wide range of topics. Based on the provided context: {context}, please provide a detailed and informative response to the following question: {question}. Draw upon your expertise to offer a comprehensive, well-reasoned, and helpful answer that addresses the query thoroughly. Your response should be tailored to the user's level of understanding and provide valuable insights , in case of out of context question ask for further Information "

    response = model.generate_content(
        [prompt],
        generation_config=generation_config,
        stream=False,
        safety_settings=harm_categories
    )
    return jsonify({"generated_content": response.text})

if __name__ == '__main__':
    app.run(debug=True)


