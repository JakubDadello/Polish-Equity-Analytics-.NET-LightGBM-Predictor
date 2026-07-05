# Stock-Exchange-Equity-Analytics-in-ML.NET

## Project Description
Developed an enterprise-grade evolution of the Stock-Exchange-Equity-Analytics system (https://github.com/JakubDadello/Stock-Exchange-Equity-Analytics.git) using C# and ML.NET to analyze financial data from 370 companies listed on the Warsaw Stock Exchange.

Reimplemented the original project using an alternative technology stack, showcasing ability to adapt architecture and tools to different environments.

The project demonstrates both technical ML skills and financial domain knowledge, with universally applicable indicators such as ROA and EBITDA, making the methodology transferable to global financial markets.

### Key Features
1. Technology Shift: Transitioned from Python to the .NET 10 ecosystem for better performance and enterprise integration.

2. Advanced Modeling: Replaced Random Forest with LightGBM (Gradient Boosting Machine) to enhance predictive accuracy on complex financial tabular data.

3. Sector-Aware Intelligence: Models financial indicators (e.g., net_profit, ROA) while accounting for the unique nature of different industry sectors.

4. Investment Grading: Automated assignment of investment rating based on robust multi-class classification.

## Repository Structure
- `StockExchangeEA.Api/` - Model Serving & Inference: ASP.NET Core Web API providing real-time investment grading predictions via REST endpoints.
- `StockExchangeEA.Schemas` - Data Contracts: Shared Class Library containing strictly typed input/output models used across the entire solution. see [Schemas.cs](StockExchangeEA.Schemas/Schemas/Schemas.cs)
- `StockExchangeEA.Trainer/` - ML Pipeline: Console application dedicated to Data Preprocessing, LightGBM Model Training, and Evaluation. see [StockExchangeEA.Trainer/README.md](StockExchangeEA.Trainer/README.md)
- `CRISP-DM.md` - detailed methodology workflow

## How to Run

To run this project locally, you can use either Docker (recommended) or the .NET SDK.


## Tech Stack

- Language & Runtime: C# 13/ .NET 10 SDK
- Machine Learning Framework: ML.NET (Microsoft.ML)
- Data Preprocessing: LINQ, Microsoft.Data.Analysis
- API Framework: ASP.NET Core
- Containerization: Docker
- Cloud Technology: Azure
- IaC: Terraform (HCL)
- Frontend/UI: TypeScript/JavaScript, HTML, CSS

Consistent with the research conducted in the Stock-Exchange-Equity-Analytics system (https://github.com/JakubDadello/Stock-Exchange-Equity-Analytics.git), this repository maintains the same data processing standards. The documentation aligns with the original project's structure but is adapted for the ML.NET ecosystem. Detailed process descriptions are available in [CRISP-DM.md](CRISP-DM.md).
