using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace PreprocessorPipeline
{
    /// <summary>
    /// Processes raw financial data from CSV. 
    /// Maintains naming consistency with ModelInput schema (snake_case from SQL).
    /// </summary>
    public static class PreprocessingSteps
    {
        public static IEstimator<ITransformer> Build(MLContext ml)
        {
            // Must match the EXACT property names in your ModelInput class
            var numericColumns = new[] { "NetIncome", "NetCashFlow", "Roe", "Roa", "Ebitda", "Cumulation" };

            return 

                // 1. Group raw numbers into a single vector named 'numeric_vector'
                ml.Transforms.Concatenate("numeric_vector", numericColumns)
        
                // 2. Impute missing values (replace NULLs/NaNs with Mean)
                .Append(ml.Transforms.ReplaceMissingValues(
                    outputColumnName: "numeric_imputed", 
                    inputColumnName: "numeric_vector", 
                    replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                
                // 3. Normalize features using Min-Max scaling to [0, 1] range
                .Append(ml.Transforms.NormalizeMinMax("numeric_scaled", "numeric_imputed"))
                
                // 4. Apply One-Hot Encoding to the categorical 'Sector' column
                .Append(ml.Transforms.Categorical.OneHotEncoding("sector_encoded", "Sector"))
                
                // 5. Final assembly: merge scaled numbers and encoded sector into 'Features'
                // This 'Features' column is what the LightGBM trainer will look for
                .Append(ml.Transforms.Concatenate("Features", "numeric_scaled", "sector_encoded"));
        }
    }
}