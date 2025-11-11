"""
Dynamic AI-Powered Entity Extraction Example
Shows how the system adapts to different content types
"""

import requests
import json

# Example 1: Research Paper
research_paper = """
Title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"

Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

Abstract: We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

Methodology: BERT uses a transformer architecture with 12 layers, 12 attention heads, and 110M parameters. The model was pre-trained on BooksCorpus (800M words) and English Wikipedia (2,500M words). We fine-tuned BERT on 11 downstream NLP tasks including GLUE, SQuAD, and SWAG.

Results: BERT achieved new state-of-the-art results on 11 NLP tasks, including 80.5% accuracy on GLUE, 93.2% F1 score on SQuAD 1.1, and 86.3% accuracy on SWAG.

Framework: TensorFlow 1.15.0, TPU training, batch size 256, learning rate 1e-4.
"""

# Example 2: Code File
python_code = """
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np

class BertClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.1):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

def train_model(model, train_loader, val_loader, epochs=10, lr=2e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_accuracy = evaluate_model(model, val_loader)
        print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

if __name__ == "__main__":
    model = BertClassifier(num_classes=3)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
"""

# Example 3: Data Analysis Notebook
data_analysis = """
# Machine Learning Model Performance Analysis

## Dataset
- Name: Customer Churn Prediction Dataset
- Size: 10,000 samples, 20 features
- Split: 70% train, 15% validation, 15% test
- Target: Binary classification (churn/no churn)

## Model Architecture
- Algorithm: Random Forest Classifier
- Parameters: 100 trees, max_depth=10, min_samples_split=5
- Features: Age, income, usage patterns, customer service calls

## Results
- Training Accuracy: 94.2%
- Validation Accuracy: 91.8%
- Test Accuracy: 90.5%
- Precision: 0.89
- Recall: 0.87
- F1-Score: 0.88

## Framework
- Python 3.8
- scikit-learn 1.0.2
- pandas, numpy for data processing
- matplotlib for visualization
"""

def test_dynamic_extraction():
    """Test the dynamic AI extraction system"""
    
    url = "http://localhost:8000/v1/extract/dynamic"
    
    examples = [
        ("Research Paper", research_paper, "research_paper.txt"),
        ("Python Code", python_code, "model.py"),
        ("Data Analysis", data_analysis, "analysis.ipynb")
    ]
    
    for name, content, file_path in examples:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        try:
            response = requests.post(url, data={
                "content": content,
                "file_path": file_path
            })
            
            if response.status_code == 200:
                result = response.json()
                
                # Display classification
                classification = result.get("classification", {})
                print(f"📋 Classification:")
                print(f"   File Type: {classification.get('file_type', 'unknown')}")
                print(f"   Domain: {classification.get('domain', 'unknown')}")
                print(f"   Intent: {classification.get('intent', 'unknown')}")
                print(f"   Focus: {classification.get('extraction_focus', 'unknown')}")
                
                # Display validation
                validation = result.get("validation", {}).get("validation_results", {})
                print(f"\n🔍 Validation:")
                print(f"   Accuracy: {validation.get('is_accurate', False)}")
                print(f"   Confidence: {validation.get('confidence_score', 0.0):.2f}")
                print(f"   Completeness: {validation.get('completeness_score', 0.0):.2f}")
                print(f"   Hallucination Risk: {validation.get('hallucination_risk', 'unknown')}")
                
                # Display extracted data
                extracted = result.get("extracted_data", {})
                print(f"\n📊 Extracted Information:")
                
                # Model info
                if "model_info" in extracted and extracted["model_info"]:
                    model_info = extracted["model_info"]
                    print(f"   Model Type: {model_info.get('type', 'N/A')}")
                    print(f"   Parameters: {model_info.get('parameters', 'N/A')}")
                    print(f"   Architecture: {model_info.get('architecture', 'N/A')[:100]}...")
                
                # Training info
                if "training_info" in extracted and extracted["training_info"]:
                    train_info = extracted["training_info"]
                    print(f"   Training: {train_info.get('is_trained', 'N/A')}")
                    print(f"   Epochs: {train_info.get('epochs', 'N/A')}")
                    print(f"   Optimizer: {train_info.get('optimizer', 'N/A')}")
                
                # Performance
                if "performance_metrics" in extracted and extracted["performance_metrics"]:
                    perf = extracted["performance_metrics"]
                    print(f"   Accuracy: {perf.get('accuracy', 'N/A')}")
                    print(f"   Loss: {perf.get('loss', 'N/A')}")
                
                # Framework
                if "framework_info" in extracted and extracted["framework_info"]:
                    fw = extracted["framework_info"]
                    print(f"   Framework: {fw.get('framework', 'N/A')}")
                    print(f"   Version: {fw.get('version', 'N/A')}")
                
                print(f"\n✅ {name} extraction completed successfully!")
                
            else:
                print(f"❌ {name} extraction failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {name} extraction error: {e}")

if __name__ == "__main__":
    print("🚀 Dynamic AI-Powered Entity Extraction Test")
    print("Make sure the API is running: uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print("\nThis system will:")
    print("1. 🤖 Analyze content to understand what it is")
    print("2. 🎯 Classify intent and determine relevant entities")
    print("3. 📊 Extract information dynamically based on content")
    print("4. ✅ Validate extraction for accuracy and completeness")
    print("5. 🚫 Detect potential hallucinations")
    
    try:
        test_dynamic_extraction()
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")
