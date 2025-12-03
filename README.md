Mitigating Gender Bias in AI-Based Hiring Systems

📌 Final Project – Group 17
👥 Team Members

Anshita Rayalla

Bala Surya Krishna Vankayala

Bhanu Prasad Cherukuvada

Harshavardhan Chary Vadla

Hemanth Reddy Sankaramaddi

Sai Pooja Sabbani

A full-stack framework designed to detect, mitigate, and explain gender-related biases in AI-assisted hiring pipelines using Large Language Models (LLMs). The project integrates modern fairness metrics, counterfactual analysis, and interpretable AI techniques to promote transparent and equitable hiring decisions.

---

## 🎯 **Overview**

This system focuses on identifying and reducing gender bias in AI-driven hiring by providing:

* **Bias Detection:** Quantitative measurement of gender-based disparities
* **Bias Mitigation:** Counterfactual augmentation + chain-of-thought (CoT) prompting
* **Explainability:** LIME-based insights and reasoning faithfulness
* **Interactive Frontend:** Real-time analysis and visualization tools

---

## 🔬 **Background & Motivation**

AI-driven hiring systems often inherit or amplify gender biases from historical recruitment data. Through structured reasoning prompts and counterfactual data augmentation, this system demonstrates meaningful reductions in gender bias while maintaining decision quality.

### **Key Findings**

* **10%+ reduction** in gender demographic parity gaps
* Improved transparency through step-by-step CoT reasoning
* Enhanced interpretability using LIME
* Minimal trade-offs in decision accuracy after mitigation

---

## 🏗️ **System Architecture**

### **1. Bias Analysis Engine**

* Computes fairness metrics (demographic parity, equalized odds, predictive parity)
* Analyzes profiles from:

  * Stack Overflow Developer Survey (73k+ samples)
  * Resume datasets (962 entries)
* Uses bootstrapped confidence intervals for statistical significance

### **2. Bias Mitigation Module**

* **Counterfactual Data Augmentation:** Creates gender-swapped parallel profiles
* **Chain-of-Thought Prompting:** Guides LLMs with structured reasoning
* **LIME Integration:** Produces interpretable explanations for each decision

### **3. Web Interface**

* Real-time candidate evaluations
* Interactive bias visualizations
* Explanation explorer for reasoning chains
* Cross-model comparison tools

---

## 🚀 **Features**

### **Bias Detection**

* Detects gender-based disparities in model predictions
* Provides statistical testing with confidence intervals
* Supports multiple LLMs (e.g., Mistral 7B, Ollama models)

### **Explainable AI**

* LIME-based feature attribution
* Reasoning faithfulness scoring
* Highlights whether decisions rely excessively on gender cues
* Provides per-candidate interpretability reports

### **Interactive Frontend**

* Bias metrics dashboard
* Candidate profile analysis
* Explanation visualization
* Model comparison view
* Exportable bias reports

---

## 🛠️ Prerequisites

* **Python 3.9+**
* **Node.js + npm/yarn**
* **Ollama** (for local LLM inference)

---

## 📊 **Usage Examples**

### **Gender Bias Analysis Request**

```python
POST /api/analyze-candidate
{
  "profile": "Software Engineer with 5 years experience in Python and ML",
  "demographics": {
    "gender": "female",
    "age": 28,
    "education": "Masters in Computer Science"
  }
}
```

### **Generating Model Explanations**

```python
POST /api/explain-decision
{
  "candidate_id": "12345",
  "explanation_type": "lime",
  "include_cot": true
}
```

---

## 📈 **Evaluation Metrics**

### **Fairness Metrics**

* **Demographic Parity:** Compares hiring rates across gender groups
* **Equalized Odds:** Compares gender-specific error rates
* **Predictive Rate Parity:** Ensures equal predictive value across genders

### **Explanation Quality**

* **Faithfulness:** Whether explanations reflect true model behavior
* **Relevance:** Alignment between reasoning and candidate data
* **Consistency:** Stability across similar candidate profiles
