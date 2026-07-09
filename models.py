from pydantic import BaseModel
from pydantic import Field

# This class mirrors C# ModelInput schema perfectly
class FinancialSchemas(BaseModel):
    NetIncome: float = Field(description="The net income or net profit of the company.")
    NetCashFlow: float = Field(description="Total net cash flow from operating, investing, and financing activities.")
    Roe: float = Field(description="Return on Equity (ROE) expressed as a percentage or decimal.")
    Roa: float = Field(description="Return on Assets (ROA) expressed as a percentage or decimal.")
    Ebitda: float = Field(description="Earnings Before Interest, Taxes, Depreciation, and Amortization.")
    Sector: str = Field(description="The industry sector. Must closely match definitions like 'technology and engineering'.")
    Cumulation: float = Field(description="Binary indicator, usually 0.0 or 1.0, regarding dividend cumulation or similar metric.")