import os
import torch
import time
from digirl.environment.android import autoui_translate_action
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import logging

def parse_action(data):
    raw_actions, labels, valids = [], [], []
    for raw_action in data:
        try:
            android_action = autoui_translate_action(raw_action)
            raw_actions.append(raw_action)
            labels.append(android_action.action_type.value)
            valids.append(1)
        except Exception as e:
            raw_actions.append(None)
            labels.append(None)
            valids.append(0)
            logging.warning(e)
            logging.warning(f"Failed to translate action: {raw_action}, terminating the environment")
    return raw_actions, labels, valids

class CausalExecutor():
    def __init__(self, tokenizer, device, causal_save_path):
        self.tokenizer = tokenizer
        self.device = device
        self.causal_save_path = causal_save_path
        if not os.path.exists(self.causal_save_path):
            os.makedirs(self.causal_save_path)
        self.time_id = str(time.time())
        self.iteration = 0
        self.softmax = torch.nn.Softmax(dim= -1)

    def get_token_effect(self, action_ids, act_labels, causal_model, window_size=4):
        self.iteration += 1
        causal_model.eval()
        special_tokens = []
        skip_chars = ['[]', '()', '{', '}', '{}', ':', '""', "'", ',', '!', '?', ';', '_', '.']
        for token_name, token in self.tokenizer.special_tokens_map.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int):
                special_tokens.append(token_id)
            elif isinstance(token_id, list):
                special_tokens += token_id
        for char in skip_chars:
            special_tokens.extend(self.tokenizer.encode(char, add_special_tokens=False))
        special_tokens = set(special_tokens)  # Convert to set for faster lookup

        act_tokens = action_ids['input_ids']  # Shape: (batch_size, seq_len)
        act_masks = action_ids['attention_mask']
        act_labels = torch.tensor(act_labels, dtype=torch.long).to(self.device)

        batch_size, seq_len = act_tokens.size()

        # Compute the model outputs on the original inputs once
        with torch.no_grad():
            original_output = causal_model(input_ids=act_tokens, attention_mask=act_masks)
            probs = self.softmax(original_output.logits)  # Shape: (batch_size, num_labels)
            # Get the probs corresponding to act_labels
            true_prob = probs.gather(1, act_labels.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)

        # Initialize the tensor to store effects
        causal_effects = torch.zeros((batch_size, seq_len), device=self.device)

        # Process positions in batches of size 32
        processing_batch_size = 64  # Adjust as per your GPU memory

        for start_pos in range(0, seq_len, processing_batch_size):
            end_pos = min(start_pos + processing_batch_size, seq_len)
            positions = torch.arange(start_pos, end_pos, device=self.device)  # Positions to process in this batch
            num_positions = positions.size(0)

            # Prepare modified inputs
            # We need to create a modified input for each position in 'positions' for each sample in the batch
            # Total modified inputs = batch_size * num_positions

            # Expand act_tokens and act_masks for the current batch of positions
            act_tokens_expanded = act_tokens.unsqueeze(1).expand(-1, num_positions, -1).contiguous().view(-1, seq_len)  # Shape: (batch_size * num_positions, seq_len)
            act_masks_expanded = act_masks.unsqueeze(1).expand(-1, num_positions, -1).contiguous().view(-1, seq_len)

            # Mask tokens at positions i to i + window_size - 1
            for offset in range(window_size):
                positions_to_mask = positions + offset
                positions_to_mask = positions_to_mask.clamp(0, seq_len - 1)
                positions_to_mask_expanded = positions_to_mask.unsqueeze(0).expand(batch_size, -1).contiguous().view(-1)  # Shape: (batch_size * num_positions,)
                # Now, mask these positions in act_tokens_expanded
                act_tokens_expanded[torch.arange(batch_size * num_positions, device=self.device), positions_to_mask_expanded] = self.tokenizer.unk_token_id

            # Compute the model outputs on the modified inputs
            with torch.no_grad():
                counterfactual_output = causal_model(input_ids=act_tokens_expanded, attention_mask=act_masks_expanded)
                counterfactual_probs = self.softmax(counterfactual_output.logits)  # Shape: (batch_size * num_positions, num_labels)
                # Get the probs corresponding to act_labels
                act_labels_expanded = act_labels.unsqueeze(1).expand(-1, num_positions).contiguous().view(-1)  # Shape: (batch_size * num_positions,)
                counterfactual_true_prob = counterfactual_probs.gather(1, act_labels_expanded.unsqueeze(1)).squeeze(1)  # Shape: (batch_size * num_positions,)

            # Expand true_probs to match batch_size * num_positions
            true_prob_expanded = true_prob.unsqueeze(1).expand(-1, num_positions).contiguous().view(-1)  # Shape: (batch_size * num_positions,)

            # Compute the effect
            effect_batch = torch.abs(true_prob_expanded - counterfactual_true_prob)  # Shape: (batch_size * num_positions,)

            # Reshape effect_batch to (batch_size, num_positions)
            effect_batch = effect_batch.view(batch_size, num_positions)

            # Store the effect at the corresponding positions
            causal_effects[:, start_pos:end_pos] = effect_batch  # Assign the effect_batch to the corresponding positions

        # Handle special tokens by setting their effects to zero
        special_tokens_mask = torch.zeros_like(act_tokens, dtype=torch.bool)
        for token in special_tokens:
            special_tokens_mask |= (act_tokens == token)
        causal_effects[special_tokens_mask] = 0.0

        # Normalize the effects across the sequence for each sample
        min_effect = causal_effects.min(dim=1, keepdim=True)[0]
        max_effect = causal_effects.max(dim=1, keepdim=True)[0]
        causal_effects = ((causal_effects - min_effect) / (max_effect - min_effect + 1e-10)).pow(0.5) # [0, 1]

        return act_tokens, causal_effects.detach()

    def render(self, act_tokens, causal_effects, num_instances=10):
        # render is called every 500 iterations
        if self.iteration % 500 != 0:
            return
        num_instances = min(num_instances, act_tokens.size(0))
        fig, ax = plt.subplots(figsize=(20, 4))  
        font_prop = FontProperties(fname=None, size=12) 
        extra_spacing = 0.0
        dpi = fig.dpi
        colors = ["#ffffff", "#cd5c5c"]  
        colormap = LinearSegmentedColormap.from_list("custom_reds", colors)
        # colormap = plt.cm.Reds

        y_offset = 0
        x_offset_max = 0
        for i in range(num_instances):
            x_offset = 0
            for j in range(act_tokens.size(1)): 
                token = self.tokenizer.decode(act_tokens[i,j])
                weight = causal_effects[i,j].item()

                text = plt.text(x_offset, y_offset, token, fontsize=10, color='black',
                                ha='left', va='center',
                                fontproperties=font_prop,
                                bbox=dict(facecolor=colormap(weight), edgecolor='none', pad=0.2))
                
                renderer = fig.canvas.get_renderer()
                bbox = text.get_window_extent(renderer=renderer)
                text_width = bbox.width / dpi  
                x_offset += (text_width + extra_spacing)

            if x_offset > x_offset_max:
                x_offset_max = x_offset
            y_offset += 1.0

        ax.set_xlim(0, x_offset_max)
        ax.set_ylim(0, y_offset - 1.0)

        ax.axis('off')
        sm = mpl.cm.ScalarMappable(cmap=colormap, norm=mpl.colors.Normalize(vmin=0, vmax=1))
        sm.set_array([]) 

        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Weight', fontsize=12)
        plt.tight_layout()

        output_path = os.path.join(self.causal_save_path, f'{self.time_id}_{self.iteration}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()