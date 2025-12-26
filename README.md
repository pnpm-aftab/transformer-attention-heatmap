# Attention Heatmap Reveal

This project visualizes the "Neural Focus" of a Large Language Model (LLM) to show how it "thinks" when processing text.

## Concept

**Attention** is the core mechanism behind Transformers (like ChatGPT, BERT, and Llama). It is simply a mathematical way for the model to weigh the importance of different parts of the input when processing a specific word.

Imagine reading a sentence: when you see the word "it", your brain instantly looks back to find what "it" refers to (e.g., "the dog"). The model does the same thing by assigning a high "attention weight" to the relevant word.

## Visualization

The generated visualization (`attention_heatmap.png`) displays a grid of **Attention Heads**.

- **Grid**: Each small square is one "Head" (a mini-brain within the layer).
- **Color Intensity**: Brighter colors mean higher attention weight.
- **Axes**: The X and Y axes show the tokens (words/parts of words) in the sentence.
- **Diagonal**: You will often see a strong diagonal line, meaning words pay attention to themselves or their immediate neighbors.
- **Vertical Stripes**: If a column is bright across many rows, it means that specific word is being "attended to" by many other words (it's important context).

## Implementation

We use the **Qwen2.5-0.5B** model, a modern and efficient LLM from late 2024/2025.

### Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the visualization:

   ```bash
   python visualize_attention.py
   ```

3. View the result:
   Open `attention_heatmap.png` to see the grid of attention patterns.

### Customization

You can modify `visualize_attention.py` to:

- Change the input text.
- Visualize a different layer (0-23).
- Use a different model (e.g., `gpt2` or `meta-llama/Llama-3.2-1B`).
