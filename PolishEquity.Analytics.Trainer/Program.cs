using Microsoft.ML;
using DataOperations;     
using Training;       

class Program
{
    public static void Main()
    {
        // 1. Initialize MLContext with a fixed seed for reproducibility
        var ml = new MLContext(seed: 42);

        // Define paths
        // In Docker, these should point to your shared volumes
        string dataPath = Path.Combine("Data", "initial_labeling_data.csv");
        string modelPath = Path.Combine("..", "PolishEquity.Analytics.Stacking", "Models", "LightGBM_model.zip");
        string stackingDataPath = Path.Combine("..", "PolishEquity.Analytics.Stacking", "Data", "stacking_input.csv");

        Console.WriteLine("Loading data...");
        // 2. Load Data from CSV
        IDataView fullData = DataService.LoadData(ml, dataPath);

        // 3. Split data (80% training, 20% test)
        var split = ml.Data.TrainTestSplit(fullData, testFraction: 0.2);
        IDataView trainingData = split.TrainSet;
        IDataView testData = split.TestSet;

        Console.WriteLine("Training LightGBM model...");
        // 4. Train the base model (C# Expert)
        ITransformer model = BoosterTrainer.Train(ml, trainingData);

        // 5. Save the trained ML.NET model
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
        ml.Model.Save(model, trainingData.Schema, modelPath);
        Console.WriteLine($"Model saved to: {modelPath}");

        // 6. Generate predictions on the test set for Stacking
        // This adds the 'Score' columns needed by the Python Meta-Learner
        IDataView predictions = model.Transform(testData);

        // 7. Save predictions to CSV
        Directory.CreateDirectory(Path.GetDirectoryName(stackingDataPath)!);

        // Wybieramy tylko konkretne kolumny, żeby Python się nie pogubił
        var selectedColumns = ml.Transforms.SelectColumns("Label", "Score");
        var selectedData = selectedColumns.Fit(predictions).Transform(predictions);

        using (var stream = new FileStream(stackingDataPath, FileMode.Create))
        {
            ml.Data.SaveAsText(selectedData, stream, separatorChar: ',', headerRow: true);
        }

        Console.WriteLine($"Stacking input CSV saved to: {stackingDataPath}");
        Console.WriteLine("Training pipeline completed successfully.");
    }
}