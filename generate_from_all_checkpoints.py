"""
Generate text samples from all checkpoints in the out directory
Output as a LaTeX table
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from tensor_parallel_model_general_order import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
out_base_dir = 'out'
start = "what did nietzsche proclaim?"  # starting prompt
num_samples = 3  # number of samples to draw per checkpoint
max_new_tokens = 100  # number of tokens generated in each sample
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# -----------------------------------------------------------------------------

def escape_latex(text):
    """Escape special LaTeX characters"""
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Find all checkpoint directories
checkpoint_dirs = []
for item in os.listdir(out_base_dir):
    item_path = os.path.join(out_base_dir, item)
    if os.path.isdir(item_path):
        ckpt_path = os.path.join(item_path, 'ckpt.pt')
        if os.path.exists(ckpt_path):
            checkpoint_dirs.append(item_path)

checkpoint_dirs.sort()  # Sort for consistent ordering

print(f"Found {len(checkpoint_dirs)} checkpoints:")
for d in checkpoint_dirs:
    print(f"  - {d}")
print("\n" + "="*80 + "\n")

# Store results: results[checkpoint_name] = [sample1, sample2, ...]
results = {}

# Load and generate from each checkpoint
for checkpoint_dir in checkpoint_dirs:
    checkpoint_name = os.path.basename(checkpoint_dir)
    print(f"Generating from: {checkpoint_name}")
    
    ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
    
    try:
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        
        # Remove unwanted prefix from state dict
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        # Try to load meta.pkl for encoding/decoding
        load_meta = False
        if 'config' in checkpoint and 'dataset' in checkpoint['config']:
            meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
            load_meta = os.path.exists(meta_path)
        
        if load_meta:
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
        else:
            # Fallback to GPT-2 tiktoken encoding (for openwebtext, etc.)
            print(f"  No meta.pkl found, using GPT-2 tiktoken encoding")
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)
        
        # Encode the prompt
        start_ids = encode(start)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        
        # Generate samples
        samples = []
        with torch.no_grad():
            with ctx:
                for k in range(num_samples):
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    generated_text = decode(y[0].tolist())
                    # Remove newlines and extra whitespace
                    generated_text = generated_text.replace('\n', ' ').replace('\r', ' ')
                    # Collapse multiple spaces into one
                    generated_text = ' '.join(generated_text.split())
                    samples.append(generated_text)
        
        results[checkpoint_name] = samples
        print(f"  Generated {len(samples)} samples")
        
        # Clean up
        del model
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results[checkpoint_name] = ["Error"] * num_samples
        continue

# Generate LaTeX table (vertical format: checkpoints as rows, samples as columns)
print("\n" + "="*80)
print("LaTeX Table Output:")
print("="*80 + "\n")

checkpoint_names = sorted(results.keys())

# Start LaTeX table with vertical layout
latex_output = []
latex_output.append(r"\begin{table}[htbp]")
latex_output.append(r"\centering")
latex_output.append(r"\small")  # Make text smaller to fit more content
latex_output.append(r"\begin{tabular}{|l|" + "p{4.5cm}|" * num_samples + "}")
latex_output.append(r"\hline")

# Header row (Sample 1, Sample 2, Sample 3...)
header = "Checkpoint & " + " & ".join([f"Sample {i+1}" for i in range(num_samples)]) + r" \\"
latex_output.append(header)
latex_output.append(r"\hline")

# Data rows (one row per checkpoint)
for checkpoint_name in checkpoint_names:
    row_cells = [escape_latex(checkpoint_name)]
    for i in range(num_samples):
        sample_text = results[checkpoint_name][i]
        # Truncate long samples for readability
        if len(sample_text) > 150:
            sample_text = sample_text[:150] + "..."
        escaped_text = escape_latex(sample_text)
        row_cells.append(escaped_text)
    
    row = " & ".join(row_cells) + r" \\"
    latex_output.append(row)
    latex_output.append(r"\hline")

latex_output.append(r"\end{tabular}")
latex_output.append(r"\caption{Generated text samples from different checkpoints. Prompt: ``" + escape_latex(start) + "''}")
latex_output.append(r"\label{tab:samples}")
latex_output.append(r"\end{table}")

# Print the LaTeX table
latex_table = "\n".join(latex_output)
print(latex_table)

# Save to file
output_file = "generated_samples_table.tex"
with open(output_file, 'w') as f:
    f.write(latex_table)

print(f"\n\nLaTeX table saved to: {output_file}")
print("="*80)

