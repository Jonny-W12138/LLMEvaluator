# Fine-grained LLMs Evaluator

## 📌 Introduction

**Fine-grained LLMs Evaluator** is a flexible and transparent evaluation framework designed to assess large language models (LLMs) from a **fine-grained capability perspective**. 

Currently, it supports comprehensive evaluation of the following core capabilities:

- ✏️ **Summarization** — Assess the model’s ability to condense and rephrase content.
- 📅 **Planning** — Evaluate how well the model generates coherent plans, sequences, or step-by-step solutions.
- 🧮 **Computation** — Test numerical calculation accuracy and quantitative reasoning.
- 🔍 **Reasoning** — Examine the model’s logical thinking and problem-solving abilities.
- 📚 **Retrieval** — Measure how effectively the model uses or extracts information from external sources or knowledge bases.

In addition to these built-in categories, you can easily define and integrate **custom capabilities** to tailor the evaluation to your specific needs.

## ⚙️ Environment Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Jonny-W12138/LLMEvaluator
cd LLMEvaluator
```

### 2️⃣ (Optional) Create a Virtual Environment

```bash
conda create -n LLMEvaluator python=3.12
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Specially, please install [BLEURT](https://github.com/google-research/bleurt) for summarization capability evaluation.

```bash
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```

## 📑 Dataset Preparation

We provide **predefined benchmark datasets** for each of the five supported capabilities: **Summarization**, **Planning**, **Computation**, **Reasoning**, and **Retrieval**. These built-in datasets help you get started quickly and ensure consistent comparisons across different models.

### 🔹 Use Built-in Datasets

By default, the evaluator comes with ready-to-use datasets for each capability. You can directly run evaluations without any additional setup.

### 🔹 Upload Custom Datasets

You can also evaluate your models with **custom datasets**:

- **Via the UI**: Use the web interface to upload new datasets.
- **Manually**: Place your dataset files in the `dataset/` directory at the project root. Organize them in subfolders (e.g., `dataset/summarization/`, `dataset/planning/`, etc.).

## ✅ Evaluation Process

The evaluation process in **Fine-grained LLMs Evaluator** is structured to guide you step-by-step, making it easy to obtain detailed insights into your model’s specific strengths and weaknesses.

### 1️⃣ **Select Evaluation Capabilities**

From the **left sidebar**, choose which capabilities you want to evaluate.
 Available capabilities include: **Summarization**, **Planning**, **Computation**, **Reasoning**, **Retrieval**, or any **custom capabilities** you have added.

### 2️⃣ **Create or Select an Evaluation Task**

After selecting capabilities, create a **new evaluation task** or reuse an **existing task**.
 Tasks define the configuration for your evaluation, including which model and dataset will be used.

### 3️⃣ **Select the Evaluation Model**

Next, choose the model to evaluate. You can pick one of two options:

- **Hugging Face** — Select a model hosted on Hugging Face and run it locally or via an inference API.
- **API Call** — Provide an API key and endpoint for a remote LLM.

### 4️⃣ **Select or Upload the Dataset**

Once the model is selected, choose the dataset to use:

- 📁 **Predefined Datasets** — Use the built-in datasets for the selected capabilities.
- ⬆️ **Upload Your Own** — Upload a custom dataset through the UI.
- ⚙️ **System-generated Data** — Alternatively, generate synthetic test cases automatically using the system’s built-in data generation tool.

### 5️⃣ **Run Fine-grained Evaluation**

After selecting the dataset, follow the on-screen instructions in the UI to run the fine-grained evaluation.
 The system will execute the tasks and collect model outputs for analysis.

### 6️⃣ **Generate an Evaluation Report**

When the evaluation is finished, you can:

- Review detailed capability scores and qualitative insights.
- Export a comprehensive evaluation report that highlights the model’s performance, strengths, and areas for improvement.
