# Gemma-2b Fine-Tuning with QLoRA

This repository contains a Google Colab notebook for fine-tuning the **Gemma-2b** base model from Google using **QLoRA (Quantized Low-Rank Adaptation)**. The model is trained on the `Abirate/english_quotes` dataset to specialize in generating and completing famous quotes.

## 🚀 Project Overview
The goal of this project is to demonstrate an efficient way to fine-tune large language models (LLMs) on consumer-grade hardware or free-tier cloud environments (like Colab) by leveraging 4-bit quantization.

### Key Features:
- **Model:** `google/gemma-2b`
- **Method:** QLoRA (4-bit quantization + LoRA adapters)
- **Library Stack:** `transformers`, `peft`, `bitsandbytes`, `trl`, and `accelerate`.
- **Dataset:** [English Quotes](https://huggingface.co/datasets/Abirate/english_quotes)
- **Monitoring:** Integrated with Weights & Biases (W&B) for training visualization.

## 🛠️ Requirements
To run this project, you need a Hugging Face account and an API token with access to the Gemma model weights.

```bash
pip install bitsandbytes peft trl accelerate datasets transformers
```

## ⚙️ Configuration
- **Quantization:** 4-bit NormalFloat (nf4) with `bfloat16` compute dtype.
- **LoRA Parameters:**
  - Rank (r): 8
  - Target Modules: All linear layers (q_proj, v_proj, etc.)
  - Task Type: Causal LM
- **Training:**
  - Batch Size: 1 (with 4 gradient accumulation steps)
  - Learning Rate: 2e-4
  - Max Steps: 100

## 📖 How to Use
1. **Setup:** Add your `HF_TOKEN` to your environment or Colab secrets.
2. **Load Model:** The notebook initializes Gemma-2b in 4-bit mode to save VRAM.
3. **Train:** Run the `SFTTrainer` block to begin the fine-tuning process.
4. **Inference:** Use the provided generation script to see the model complete quotes like:
   - *Input:* "Quote: Imagination is more"
   - *Output:* "important than knowledge..."

## 📊 Results
The fine-tuned model shows improved adherence to the "Quote: [Text] Author: [Name]" format and successfully recalls/generates famous literary and historical quotes.
