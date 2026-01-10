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
            var pipeline = PreprocessingSteps.Build(ml)
                .Append(ml.MulticlassClassification.Trainers.LightGbm(MultiClassOptions))
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model
            var model = pipeline.Fit(data);

            // Evaluate the model on the same dataset (or use a split if needed)
            var predictions = model.Transform(data);
            var metrics = ml.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            Console.WriteLine("===== MODEL METRICS =====");

            // Accuracy
            Console.WriteLine($"Accuracy (Micro): {metrics.MicroAccuracy:F4}");
            Console.WriteLine($"Accuracy (Macro): {metrics.MacroAccuracy:F4}");

            // Extract confusion matrix
            var matrix = metrics.ConfusionMatrix;
            int classes = matrix.NumberOfClasses;

            Console.WriteLine("\n===== PRECISION & RECALL PER CLASS =====");

            // Compute precision and recall manually for each class
            for (int i = 0; i < classes; i++)
            {
                double tp = matrix.Counts[i][i];
                double fp = 0;
                double fn = 0;

                // False Positives: column sum except TP
                for (int row = 0; row < classes; row++)
                    if (row != i)
                        fp += matrix.Counts[row][i];

                // False Negatives: row sum except TP
                for (int col = 0; col < classes; col++)
                    if (col != i)
                        fn += matrix.Counts[i][col];

                double precision = tp / (tp + fp + 1e-6f);
                double recall = tp / (tp + fn + 1e-6f);

                Console.WriteLine($"Class {i}: Precision={precision:F4}, Recall={recall:F4}");
            }

            // Print confusion matrix
            Console.WriteLine("\n===== CONFUSION MATRIX =====");
            for (int i = 0; i < classes; i++)
            {
                Console.Write($"Class {i}: ");
                for (int j = 0; j < classes; j++)
                {
                    Console.Write($"{matrix.Counts[i][j]} ");
                }
                Console.WriteLine();
            }

            Console.WriteLine("============================");

            return model;
        }
    }   
}