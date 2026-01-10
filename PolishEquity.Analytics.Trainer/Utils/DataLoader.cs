using Microsoft.ML;
using Schemas;

namespace MLProject.Data
{
    public static class DataService
    {
        /// <summary>
        /// Ingests data from CSV. Uses lazy loading via IDataView to optimize memory footprint.
        /// </summary>
        public static IDataView LoadData(MLContext ml, string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("Dataset file missing.", path);

            // Schema is automatically mapped based on ModelInput attributes
            return ml.Data.LoadFromTextFile<ModelInput>(
                path: path,
                hasHeader: true,
                separatorChar: ','
            );
        }
    }
}

