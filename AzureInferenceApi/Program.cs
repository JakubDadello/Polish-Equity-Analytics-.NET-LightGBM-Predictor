using Microsoft.Extensions.ML; //provide access to the PredictionEnginePool
using Schemas; //provide access to the input and output schemas  

//create a configuration object for the web application
var builder = WebApplication.CreateBuilder(args);

//register the PredictionEnginePool (a pool includes a few predictors)
builder.Services.AddPredictionEnginePool<ModelInput, ModelOutput>()
    .FromFile("models/LightGBM_model.zip"); //load the Light GBM model 

//build the web application
var app = builder.Build();

//define the endpoint
app.MapPost("/predict", (PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool, ModelInput input) =>
{

    //validation
    if (input == null)
    {
        return Results.BadRequest("Input data cannot be null.");
    }

    //prediction
    ModelOutput prediction = predictionEnginePool.Predict(input);

    //return a result
    return Results.Ok(prediction);
});

//run the web application
app.Run();