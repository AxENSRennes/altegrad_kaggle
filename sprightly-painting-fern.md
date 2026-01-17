# Add Thinking Mode Support for Stage 2 Training

## Goal

Add optional thinking mode for Stage 2 SFT training where:
- **Training**: Model generates `<think>...</think>` freely, loss computed only on description after `</think>`
- **Inference**: Model reasons in `<think>...</think>` before generating description

## Key Insight: Dynamic Loss Masking

We don't include thinking tokens in training labels. Instead:
1. Training input: `[PROMPT]` (no thinking structure)
2. Training target: `[DESCRIPTION]<eos>` (just description)
3. Model generates: `<think>...</think>[predicted_desc]<eos>`
4. **Dynamic masking**: Find `</think>` in output, compute loss only after it

This is more complex but allows truly free thinking without constraining the structure.

## Architecture

```
Training:
┌─────────────────────────────────────────────────────────────────────┐
│ Input: [PROMPT without /no_think]                                   │
│                                                                     │
│ Model output (logits): <think> ... </think> [predicted_desc] <eos>  │
│                        │<── ignore ──>│ │<── compute loss ──>│      │
│                                                                     │
│ Target labels: [DESCRIPTION] <eos>                                  │
│                │<── align here ──>│                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Files to Modify

| File | Change |
|------|--------|
| `config.py` | Add `SYSTEM_PROMPT_THINK` |
| `dataset_caption.py` | Add `enable_thinking` param, simpler target (just description) |
| `train_stage2.py` | **Custom loss** with dynamic `</think>` detection |
| `model_wrapper.py` | Add `enable_thinking` to `generate()` |
| `run.py` | Add `--thinking` CLI flag |

**Note**: `data_collator.py` changes minimally - no thinking_length needed.

## Implementation

### 1. `config.py` - Add thinking prompt (~line 218)

```python
# Existing
SYSTEM_PROMPT = "/no_think\nYou are an expert chemist. Describe the molecule's chemical properties and functional groups concisely."

# New
SYSTEM_PROMPT_THINK = "You are an expert chemist. Describe the molecule's chemical properties and functional groups concisely."

USER_PROMPT_FORMAT = "Molecule Structure: <|graph|>\nSMILES: {smiles}"
```

### 2. `dataset_caption.py` - Simpler target for thinking mode

```python
class MolCaptionDatasetTRL:
    def __init__(self, ..., enable_thinking: bool = False):
        ...
        self.enable_thinking = enable_thinking

    def _build_prompt(self, smiles: str) -> str:
        system_prompt = SYSTEM_PROMPT_THINK if self.enable_thinking else SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPT_FORMAT.format(smiles=smiles)}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # No <think> tags from template
        )
        return prompt

    def __getitem__(self, idx):
        ...
        prompt = self._build_prompt(smiles)

        # For thinking mode: target is just description (loss computed dynamically)
        # For no-think mode: keep existing behavior
        full_text = prompt + description + self.tokenizer.eos_token

        return {
            'graph': graph,
            'text': full_text,
            'prompt_length': len(self.tokenizer.encode(prompt, add_special_tokens=False)),
            'enable_thinking': self.enable_thinking,  # NEW: pass flag for custom loss
            ...
        }
```

### 3. `train_stage2.py` - Custom loss with dynamic masking

This is the core change. In `MolCaptionTrainer.compute_loss()`:

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    graphs = inputs.get("graphs")
    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")
    labels = inputs.get("labels")  # Pre-masked labels (prompt masked)
    enable_thinking = inputs.get("enable_thinking", False)

    # Forward pass
    outputs = self.mol_model(
        graphs=graphs,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,  # Don't use HF's loss - we compute custom
    )
    logits = outputs["logits"]

    if enable_thinking:
        # Dynamic loss: find </think> in predicted sequence, mask everything before
        loss = self._compute_thinking_loss(logits, labels, input_ids)
    else:
        # Standard loss with pre-masked labels
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

    return (loss, outputs) if return_outputs else loss

def _compute_thinking_loss(self, logits, labels, input_ids):
    """
    Compute loss only on tokens after </think> in model output.

    Args:
        logits: [batch, seq_len, vocab_size] - model predictions
        labels: [batch, seq_len] - target with prompt masked (-100)
        input_ids: [batch, seq_len] - input token ids
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    # Get </think> token ID
    think_end_id = self.mol_model.tokenizer.encode("</think>", add_special_tokens=False)
    # Usually this is a multi-token sequence, get the last token
    think_end_token = think_end_id[-1] if think_end_id else None

    if think_end_token is None:
        # Fallback to standard loss if can't find token
        return F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)

    # Get predicted tokens (argmax)
    pred_ids = logits.argmax(dim=-1)  # [batch, seq_len]

    # Create dynamic mask
    dynamic_labels = labels.clone()

    for i in range(batch_size):
        # Find </think> position in predicted sequence
        think_positions = (pred_ids[i] == think_end_token).nonzero(as_tuple=True)[0]

        if len(think_positions) > 0:
            # Found </think> - mask everything up to and including it
            think_end_pos = think_positions[0].item()

            # Find where actual content starts (after padding)
            content_mask = labels[i] != -100
            if content_mask.any():
                content_start = content_mask.nonzero(as_tuple=True)[0][0].item()

                # If </think> is within the content region, mask up to it
                if think_end_pos >= content_start:
                    dynamic_labels[i, content_start:think_end_pos + 1] = -100

    # Compute loss with dynamic mask
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        dynamic_labels.view(-1),
        ignore_index=-100
    )

    return loss
```

### 4. `data_collator.py` - Pass thinking flag

Add `enable_thinking` to the batch output:

```python
def __call__(self, features):
    ...
    # Check if any sample has thinking enabled (all should be same in a batch)
    enable_thinking = features[0].get('enable_thinking', False)

    return {
        'graphs': batched_graphs,
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels,
        'enable_thinking': enable_thinking,  # NEW
    }
```

### 5. `model_wrapper.py` - Update generate()

```python
def generate(self, graphs, smiles_list, ..., enable_thinking: bool = False):
    system_prompt = SYSTEM_PROMPT_THINK if enable_thinking else SYSTEM_PROMPT
    ...
    prompt = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    ...
    output_text = self.tokenizer.decode(...)

    if enable_thinking and "</think>" in output_text:
        output_text = output_text.split("</think>", 1)[1].strip()

    return output_text
```

### 6. `run.py` - Add CLI flag

```python
parser.add_argument("--thinking", action="store_true",
                    help="Enable thinking mode (model reasons before answering)")

# Pass to dataset and trainer
```

## Expected Behavior

### Training (thinking mode):

```
Epoch 1:
  Model sees: [PROMPT without /no_think]
  Model generates: "<think>hmm let me think...</think>some desc"
  Loss computed on: "some desc" vs "real description"

Epoch N:
  Model learns: after </think>, output accurate description
  Thinking content: completely free, no supervision
```

### Inference (thinking mode):

```
Input: [PROMPT]
Output: "<think>
This molecule has SMILES CCO which indicates an ethanol...
The hydroxyl group suggests alcohol properties...
</think>

This molecule is ethanol, a simple alcohol with a hydroxyl group."

Extracted: "This molecule is ethanol, a simple alcohol with a hydroxyl group."
```

## Usage

```bash
# Current behavior - unchanged
python run.py --mode quick

# Thinking mode
python run.py --mode quick --thinking

# Inference
python run.py --inference --checkpoint outputs/stage2_quick_best.pt --thinking
```

## Verification

```bash
# 1. Existing tests pass
pytest tests/test_chat_template.py -v

# 2. Verify </think> token ID
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
print('</think> tokens:', tok.encode('</think>', add_special_tokens=False))
"

# 3. Test dynamic loss computation (unit test)
# Verify that loss is only computed on description tokens
```

## Caveats

1. **Gradient through argmax**: The `pred_ids = logits.argmax()` operation is not differentiable. The mask is computed from hard predictions, which may cause training instability. Alternative: use soft attention or Gumbel-softmax.

2. **Variable output length**: Model may generate very long or very short thinking. Need to handle edge cases (no `</think>` found, `</think>` at end of sequence).

3. **Training efficiency**: Custom loss is slower than built-in HF loss.

## Alternative: Soft masking (if needed)

If hard argmax causes issues, we could use soft masking:
```python
# Soft probability of each position being after </think>
think_end_probs = F.softmax(logits[:, :, think_end_token], dim=-1)
# Cumulative probability
cumsum = think_end_probs.cumsum(dim=-1)
# Soft mask: weight loss by probability of being after </think>
weights = cumsum.detach()  # Don't backprop through mask
loss = (F.cross_entropy(..., reduction='none') * weights).mean()
```

## Sources

- [Qwen3 HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Qwen3 Chat Template](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)
- [Custom Loss in Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.compute_loss)
