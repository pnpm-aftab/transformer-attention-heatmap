# Interactive Attention Heatmap

This tool provides an interactive visualization of the attention mechanisms in transformer-based models like Qwen or GPT-2. It extracts attention weights across all layers and heads to show how a model relates tokens within a given text. Users can explore attention matrices, head-specific entropy, and token distributions through a dynamic web dashboard. The project aims to provide researchers with a deep, intuitive understanding of neural focus in modern LLMs.

## How to Use

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Visualization**:
   ```bash
   python visualize_interactive.py
   ```

3. **View the Results**:
   Open the generated `attention_interactive.html` file in any modern web browser to explore the model's attention patterns.

## Customization

You can modify the `visualize_interactive()` call at the bottom of `visualize_interactive.py` to:
- Change the **model** (e.g., `"gpt2"` or `"Qwen/Qwen2.5-0.5B"`).
- Change the **input text** to analyze different sentences and prompts.
