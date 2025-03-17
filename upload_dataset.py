import requests
import uuid
import dotenv
from pathlib import Path

from langsmith import Client
from langsmith.schemas import ExampleUploadWithAttachments, Attachment

dotenv.load_dotenv()

# # Publicly available test files
# pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
wav_url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"


# # Fetch the files as bytes

# pdf_bytes = requests.get(pdf_url).content
wav_bytes = requests.get(wav_url).content


# Define the LANGCHAIN_API_KEY environment variable with your API key
langsmith_client = Client()

dataset_name = "llm-eval-demo"

dataset = langsmith_client.create_dataset(
    dataset_name=dataset_name,
    description="Test dataset for evals with publicly available attachments",
)

# Creating invoice examples
invoice_count = 3
for i in range(3):
    # Create example id
    example_id = uuid.uuid4()

    # Define the example with attachments
    example = ExampleUploadWithAttachments(
        id=example_id,
        inputs={
            "question_1": "What is the VAT registration number?",
            "question_2": "What is the invoice date? Reply in YYYY-MM-DD format.",
            "question_3": "What is the due date for payment? Reply in YYYY-MM-DD format.",
            "question_4": "How many items are there in the invoice? Please provide a numeric answer.",
            "question_5": "Extract the domain from the supplier's email address.",
        },
        outputs={
            "answer_1": ["0987654321", "1597536284", "654789321"][i],
            "answer_2": ["2024-02-06", "2024-02-06", "2025-01-31"][i],
            "answer_3": ["2024-03-07", "2025-03-07", "2025-03-07"][i],
            "answer_4": ["1", "3", "3"][i],
            "answer_5": ["vendor.com", "vendor2.com", "vendor3.com"][i],
        },
        attachments={
            "file": (
                "image/jpg",
                Path(__file__).parent / f"data/invoice_{i+1}.jpg",
            ),
        },
    )

    # Upload the examples with attachments
    # Must pass the dangerously_allow_filesystem flag to allow file paths
    langsmith_client.upload_examples_multipart(
        dataset_id=dataset.id, uploads=[example], dangerously_allow_filesystem=True
    )

# Creating charts examples

# Create example id
example_id = uuid.uuid4()

# Define the example with attachments
example = ExampleUploadWithAttachments(
    id=example_id,
    inputs={
        "question_1": "Which event happened in 2021 according to the chart? A) Hardware Offer Launch\nB) Apple One Launch\nC) Price Change\nD) No event indicated",
        "question_2": "Which year does the chart show the Price Change occurred? A) 2022\nB) 2023\nC) 2024\nD) No year specified",
        "question_3": "Which event is marked in 2023 on the chart? A) Price Change\nB) Apple One Launch\nC) Hardware Offer Launch\nD) Subscription Tier Removal",
        "question_4": "What is the overall trend of the subscriber growth from 2020 to 2024? A) A steady increase over time\nB) A sharp drop after 2021\nC) Completely flat with no changes\nD) Sporadic increases and decreases",
        "question_5": "According to the chart, in which year did the subscriber count exceed its 2020 level? A) It never exceeded 2020\nB) 2021\nC) 2022\nD) 2024",
    },
    outputs={
        "answer_1": "B",
        "answer_2": "B",
        "answer_3": "C",
        "answer_4": "A",
        "answer_5": "B",
    },
    attachments={
        "file": ("image/png", Path(__file__).parent / "data/apple_chart.png"),
    },
)
langsmith_client.upload_examples_multipart(
    dataset_id=dataset.id, uploads=[example], dangerously_allow_filesystem=True
)

print("Dataset uploaded")
