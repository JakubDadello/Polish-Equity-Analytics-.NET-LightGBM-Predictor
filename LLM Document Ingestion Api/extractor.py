import openai
import instructor
from models import FinancialSchemas

# Initialize the Instructor client by patching the official OpenAI client
# This injects Pydantic validation capabilities directly into the OpenAI SDK
client = instructor.from_openai(openai.OpenAI())

# The raw text extracted from the uploaded document
document_text = "Our company achieved an EBITDA of 250000.50 USD this year. Net income hit 150000.00... ROE is 14.5%..."

# response is guaranteed to be an instance of FinancialSchemas
response = client.chat.completions.create(

    model="gpt-4o",
    response_model = FinancialSchemas,    #this parameter provides Structured Outputs
    max_tokens=1000, #restricts the maximum response length to optimize cost and performance
    messages=[
            {"role": "user", "content": f"Extract financial metrics from this text: {document_text}"}
        ],
)
