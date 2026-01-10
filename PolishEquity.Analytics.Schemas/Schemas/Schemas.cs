using Microsoft.ML.Data;

namespace Schemas 
{
    /// <summary>
    /// Represents the raw input schema for the financial health model.
    /// Mapping is based on the CSV column index.
    /// </summary>
    public class ModelInput 
    {
        [LoadColumn(0)] public float NetIncome { get; set; }
        [LoadColumn(1)] public float NetCashFlow { get; set; }
        [LoadColumn(2)] public float Roe { get; set; }
        [LoadColumn(3)] public float Roa { get; set; }
        [LoadColumn(4)] public float Ebitda { get; set; }
        
        // Categorical feature - requires OneHotEncoding or FeaturizeText in the pipeline
        [LoadColumn(5)] public string Sector { get; set; } = "";
        
        [LoadColumn(6)] public float Cumulation { get; set; }

        // The target column we want to predict
        [LoadColumn(7)]
        [ColumnName("Label")]
        public string InvestmentAssessment { get; set; } = "";
    }

    /// <summary>
    /// Contains the prediction result. 
    /// ColumnName attribute matches the internal ML.NET prediction column name.
    /// </summary>
    public class ModelOutput
    {
        // "Score" is the default for Regression, "PredictedLabel" for Classification
        [ColumnName("PredictedLabel")] 
        public string InvestmentAssessment { get; set; } = "";
        
        public float[] Score { get; set; } = Array.Empty<float>();
    }
}