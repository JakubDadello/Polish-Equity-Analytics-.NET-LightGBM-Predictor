using Microsoft.ML;
using Microsoft.ML.Trainers.LightGbm;
using PreprocessorPipeline;


namespace Training { 
    public static class BoosterTrainer
    {
        // LightGBM configuration for multiclass classification
        public static readonly LightGbmMulticlassTrainer.Options MultiClassOptions =
            new LightGbmMulticlassTrainer.Options
            {
                NumberOfLeaves = 50,
                MinimumExampleCountPerLeaf = 20,
                LearningRate = 0.01,
                NumberOfIterations = 200,
                LabelColumnName = "Label",
                FeatureColumnName = "Features"
            };

            public static ITransformer Train(MLContext ml, IDataView data)
            {
                // Build preprocessing + LightGBM training pipeline
                // 1. Map the text 'InvestmentAssessment' to numeric 'Label' keys
                var pipeline = ml.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "Label", 
                    inputColumnName: "Label"
                ) 

                // 2. Append the feature engineering steps 
                .Append(PreprocessingSteps.Build(ml)) 

                // 3. Add the LightGBM multiclass trainer
                .Append(ml.MulticlassClassification.Trainers.LightGbm(MultiClassOptions))

                // 4. Convert the numeric 'PredictedLabel' back to the original text value
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model
            var model = pipeline.Fit(data);

            Console.WriteLine("Model training completed successfully.");

            return model;
        }
    }   
}