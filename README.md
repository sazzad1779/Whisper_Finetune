<h1>Whisper_Finetune</h1>

### Whisper Model Fine-Tuning Pipeline

This repository contains a streamlined pipeline for fine-tuning OpenAI's Whisper model on custom audio datasets. The pipeline has been designed for flexibility, efficiency, and ease of use, focusing on key steps required to train and evaluate a speech-to-text system.

<img src="assets/whisper.jpg" alt="Whisper_Finetune steps" width="600" style="border-radius: 10px;" />

---

#### Pipeline Overview

The fine-tuning pipeline consists of the following steps:

1. **Select Dataset**: Choose a dataset containing audio files and corresponding transcriptions for training.
2. **Load Dataset**: Load the dataset into a compatible format for preprocessing.
3. **Load Whisper Tools**: Initialize `WhisperFeatureExtractor`, `WhisperTokenizer`, and `WhisperProcessor`.
4. **Process Dataset**:
   - Resample audio data to 16 kHz.
   - Convert audio to log-Mel spectrograms.
   - Encode transcriptions into token IDs.
5. **Training and Evaluation**:
   - **DataLoader Creation**: Prepare the dataset for efficient batching.
   - **Metrics Initialization**: Use Word Error Rate (WER) as the evaluation metric.
   - **Load Model**: Load the pre-trained Whisper model for fine-tuning.
   - **Define Training Configurations**: Use `Seq2SeqTrainer` for training with `Seq2SeqTrainingArguments`.
6. **Generate Response**: Evaluate the fine-tuned model on test data and generate predictions.

---

#### Data Processing Details

The dataset is processed in the following steps:
1. Convert all audio files to 16 kHz.
2. Resample audio data.
3. Compute log-Mel spectrograms for model input.
4. Encode transcriptions into label IDs compatible with the tokenizer.

---

#### Prerequisites

- Python 3.8 or later
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- `datasets` library for dataset loading
- GPU with CUDA support (recommended)

Install the required dependencies using:

```bash
pip install transformers datasets librosa evaluate jiwer gradio
```

---
### How to Use
#### Step 1: Clone the Repository
```
git clone https://github.com/sazzad1779/Whisper_Finetune.git
cd Whisper_Finetune
```
#### Step 2: Prepare the Dataset
Ensure your dataset is structured with audio files and corresponding transcriptions. Update the script to point to your dataset directory or Hugging Face dataset.

#### Step 3: Run the Pipeline
Modify and execute the training script:

``
python train.py
``
#### Step 4: Evaluate the Model
Once training is complete, run the evaluation script to calculate Word Error Rate (WER):

``
python evaluate.py
``
---
### Results
- The fine-tuned model achieves improved transcription accuracy on the custom dataset.
- Word Error Rate (WER) is used as the primary evaluation metric.
### Future Work
Add data augmentation for robustness.
Experiment with advanced evaluation metrics.
Extend the pipeline to handle multimodal tasks.


Here are five brief examples of life from different perspectives:1. **Microbial Life**: *Example*: Bacteria found in the human gut, which helps digest food. *Relatability*: Just like how these microbes help us process our meals, AI systems can aid in processing complex data.2. **Plant Life**: *Example*: A sprouting seedling breaking through soil, symbolizing growth and renewal. *Relatability*: Just as a plant needs water, sunlight, and air to thrive, AI requires specific parameters, computational power, and strategic planning for optimal performance.3. **Digital Life**: *Example*: An autonomous drone navigating through obstacles with precision. *Relatability*: Much like an autonomous drone adapting its path in response to new data, machine learning systems update their decision-making based on the data they process.4. **Ecosystem Life**: *Example*: Coral reefs providing habitat for a diverse array of marine life. *Relatability*: Just as coral reefs provide multiple species with home and sustenance, blockchain provides numerous applications across various industries, serving different needs.5. **Human Life**: *Example*: An artist expressing themselves through music or paint, showcasing individuality. *Relatability*: Like how a canvas brings out the creativity of an artist, data science empowers organizations to extract insights from their information reserves, inspiring innovation.Which one resonates with you the most? Or feel free to add any topic you'd like me to explore!


It seems like I'm in the middle of a conversation, but there was no initial conversation or question to respond to. Since you've asked for a "full code example" without specifying what it's for, I'll provide some general guidance on how to approach this and offer a few examples. Please feel free to specify your needs:**General Guidance:**1. **Identify the Problem**: Clearly define the problem or task you want the code to address (e.g., text classification, image processing, etc.).2. **Choose the Language**: Select a programming language based on your needs and proficiency level.3. **Select Libraries/Frameworks**: Pick relevant libraries or frameworks that make implementing your solution easier.**Example Requests and Responses:**Since you've not specified a particular area (e.g., blockchain, machine learning), I'll provide examples for common requests:### **Python Example for Beginners**```python# A simple example to get started with Pythondef greet(name): """Greeting function.""" return f"Hello, {name}!"# Usageprint(greet("Athena2")) # Output: Hello, Athena2!```### **Python Example for Data Science (Basic Data Analysis)**```pythonimport pandas as pd# Load a sample datasetdata = { 'Name': ['John', 'Mary', 'David'], 'Age': [25,31,21]}df = pd.DataFrame(data)# Basic analysis: Count by age groupprint(df.groupby('Age').count())```### **Basic Blockchain Concept (Python, Simplified Example)**```pythonclass Block: def __init__(self): self.transaction = "" def add_transaction(self, transaction): self.transaction += transaction# Creating a simple blockchainblockchain = []def add_block(transaction): global blockchain new_block = Block() new_block.add_transaction(transaction) blockchain.append(new_block)# Usageadd_block("Transaction1")print(blockchain[0].transaction) # Output: Transaction1```**Please Specify Your Needs**: To provide a more accurate and relevant code example, I need to know:- **What task or problem do you want the code to solve?**- **Which programming language or framework do you prefer?**- **Any specific libraries or tools you'd like it to use?**Feel free to ask anything about these examples or to specify your needs!
