// declare a type for possible results of prediction 
type PredictionResult = {
    score: number[];
};

// Global state to store the file uploaded by the user
let selectedFile: File | null = null;

const fileInput = document.getElementById("file-input") as HTMLInputElement;
const dropZone = document.getElementById("drop-zone") as HTMLElement;
const fileNameDisplay = document.getElementById("file-name") as HTMLElement;
const analyzeBtn = document.getElementById("btn-generate-report") as HTMLButtonElement;

// Handle file input changes via file browser
fileInput?.addEventListener("change", (e) => {
    const target = e.target as HTMLInputElement;
    if (target.files && target.files.length > 0) {
        handleFileSelection(target.files[0]);
    }
});

// Drag and drop event listeners to enhance UX
dropZone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone?.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone?.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
    }
});

function handleFileSelection(file: File) {
    selectedFile = file;
    if (fileNameDisplay) {
        fileNameDisplay.textContent = `Selected: ${file.name}`;
    }
}

// Main orchestration flow triggered by the analysis button
analyzeBtn?.addEventListener("click", async () => {
    if (!selectedFile) {
        alert("Please upload a financial report file first.");
        return;
    }

    try {
        analyzeBtn.disabled = true;
        
        // Step 1: Send the unstructured document to the LLM for structured data extraction
        const extractedData = await extractMetricsWithAI(selectedFile);
        
        // Step 2: Pass the strongly-typed JSON structure directly to the ML.NET 10 model in Azure ACI
        await predict(extractedData);

    } catch (error) {
        console.error("Analysis sequence failed:", error);
    } finally {
        analyzeBtn.disabled = false;
    }
});

// Communicates with the LLM parsing gateway using Structured Outputs (JSON format)
async function extractMetricsWithAI(file: File): Promise<any> {
    const formData = new FormData();
    formData.append("file", file);

    const LLM_SERVICE_URL = (import.meta as any).env.VITE_LLM_SERVICE_URL;
    if (!LLM_SERVICE_URL) {
        throw new Error("LLM Service environment variable is missing");
    }

    const response = await fetch(`${LLM_SERVICE_URL}/extract`, {
        method: "POST",
        body: formData
    });

    if (!response.ok) throw new Error(`AI extraction failed with status: ${response.status}`);
    
    // The LLM gateway maps text data to matching C# property schemas (e.g., NetIncome, Roe, Sector)
    return await response.json();
}

// Submits extracted features to the ML.NET inference engine hosted on Azure Container Instances
async function predict(financialData: any): Promise<void> {
    const API_URL = (import.meta as any).env.VITE_ENDPOINT_SERVICE;
    if (!API_URL) {
        throw new Error("Inference API endpoint environment variable is missing");
    }

    const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(financialData)
    });

    if (!res.ok) throw new Error(`ML engine inference failed with status: ${res.status}`);

    const predictionOutput = await res.json();
    
    // ML.NET multiclass pipeline returns probabilities inside the 'score' array
    if (predictionOutput && predictionOutput.score) {
        updateProgressBars(predictionOutput.score);
    }
}

// Smoothly scales UI progress bars using computed multiclass probability margins
function updateProgressBars(score: number[]) {

    const high = Math.round((score[0] || 0) * 100);
    const middle = Math.round((score[1] || 0) * 100);
    const low = Math.round((score[2] || 0) * 100);

    const highEl = document.getElementById("fill-high");
    const middleEl = document.getElementById("fill-middle");
    const lowEl = document.getElementById("fill-low");

    if (highEl) { 
        highEl.style.width = `${high}%`; 
        highEl.textContent = `${high}%`; 
    }

    if (middleEl) {
        middleEl.style.width = `${middle}%`; 
        middleEl.textContent = `${middle}%`; 
    }

    if (lowEl)  { 
       lowEl.style.width = `${low}%`; 
       lowEl.textContent = `${low}%`; 
    }
}