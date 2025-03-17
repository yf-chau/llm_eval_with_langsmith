import os
import dotenv
from langsmith import traceable, Client

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import base64
import httpx

dotenv.load_dotenv()

# Initialise Gemini 2.0
gemini_flash = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
gemini_msg = gemini_flash.invoke(messages)
print("Gemini Flash 2.0")
print(gemini_msg.content)

# Initialise GPT4o with code interpreter

gpt4o = AzureChatOpenAI(
    deployment_name=os.environ["AZURE_DEPLOYMENT_NAME"],
)

gpt4o_msg = gpt4o.invoke(messages)
print("GPT 4o")
print(gpt4o_msg.content)


# Optionally add the 'traceable' decorator to trace the inputs/outputs of this function.
@traceable
def my_app(inputs: dict, attachments: dict, model_name: str = "gpt-4o") -> dict:
    instruction = "You are a helpful assistant. The user will ask you a question regarding a file attached. Answer the question as concise as possible. If the question is a multiple choice question, answer with the letter 'A', 'B', 'C' or 'D' only."

    is_image, is_pdf = False, False
    question_list = []
    for key, value in inputs.items():
        if key.startswith("question_"):
            question_list.append((value))

    if "file" in attachments:
        is_image = True
        url = attachments["file"]["presigned_url"]
        response = httpx.get(url)
        data = base64.b64encode(response.content).decode("utf-8")

    outputs = {}
    for index, question in enumerate(question_list):
        system_message = SystemMessage(
            content=instruction,
        )
        if is_image or is_pdf:
            human_message = HumanMessage(
                content=[
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{data}"},
                    },
                ]
            )
        else:
            human_message = HumanMessage(content=[{"type": "text", "text": question}])

        if model == "gemini-2.0-flash":
            ai_msg = gemini_flash.invoke([system_message, human_message])
        else:
            ai_msg = gpt4o.invoke([system_message, human_message])
        outputs[f"answer_{index+1}"] = ai_msg.content

    return outputs


ls_client = Client()
dataset_name = "llm-eval-demo"
dataset = ls_client.read_dataset(dataset_name=dataset_name)


def correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    print("Outputs")
    print(outputs)
    print("------------------")
    print("Reference Outputs")
    print(reference_outputs)
    print("------------------")

    results = []
    for key, value in outputs.items():
        if key not in reference_outputs or value != reference_outputs[key]:
            results.append({"key": key, "score": False})
        else:
            results.append({"key": key, "score": True})

    return results


# Can equivalently use the 'evaluate' function directly:
# from langsmith import evaluate; evaluate(...)

model_list = ["gpt-4o", "gemini-2.0-flash"]


for model in model_list:

    def app_with_model(inputs, attachments):
        return my_app(inputs, attachments, model_name=model)

    results = ls_client.evaluate(
        app_with_model,
        data=dataset_name,
        evaluators=[correct],
        experiment_prefix=model,  # optional, experiment name prefix
        description=f"Evaluating {model} performace",  # optional, experiment description
        max_concurrency=4,  # optional, add concurrency
    )
