using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace PreprocessorPipeline
{
    /// <summary>
    /// Processes raw financial data from CSV. 
    /// Maintains naming consistency with ModelInput schema (snake_case).
    /// </summary>
    public static class PreprocessingSteps
    {
        public static IEstimator<ITransformer> Build(MLContext ml)
        {
            // Must match the EXACT names from your CSV / ModelInput class
            var numericColumns = new[] { "NetIncome", "NetCashFlow", "Roe", "Roa", "Ebitda", "Cumulation" };

            return ml.Transforms
                // 1. Group raw numbers into a single vector
                .Concatenate("numeric_vector", numericColumns)
                
                // 2. Impute missing values with median
                // Input: numeric_vector -> Output: numeric_imputed
                .Append(ml.Transforms.ReplaceMissingValues(
                    outputColumnName: "numeric_imputed", 
                    inputColumnName: "numeric_vector", 
                    replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                
                // 3. Standardize features to zero mean and unit variance
                .Append(ml.Transforms.NormalizeMinMax("numeric_scaled", "numeric_imputed"))
                
                // 4.  Apply One-Hot Encoding (ignore unknown categories in test set)
                .Append(ml.Transforms.Categorical.OneHotEncoding("sector_encoded", "Sector"))
                
                // 5. Final assembly into the mandatory 'Features' column
                .Append(ml.Transforms.Concatenate("Features", "numeric_scaled", "sector_encoded"));
        }
    }
}