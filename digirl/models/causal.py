from transformers import (
    PreTrainedModel,
    BertConfig
    )
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch import nn

class MiniBertConfig(BertConfig):
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=3,
        num_labels=2,
        intermediate_size=512, 
        hidden_act='gelu',
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            **kwargs
        )
        self.num_labels = num_labels

class MiniBertForSequenceClassification(PreTrainedModel):
    config_class = MiniBertConfig

    def __init__(self, config, accelerator):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.accelerator = accelerator

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size, 
                nhead=config.num_attention_heads, 
                dim_feedforward=config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                activation=config.hidden_act
            ) for _ in range(config.num_hidden_layers)
        ])

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        x = self.embeddings(input_ids)  # (batch_size, seq_length, hidden_size)
        
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        x = x + position_embeddings + token_type_embeddings
        x = self.LayerNorm(x)
        x = self.dropout(x)
        
        x = x.transpose(0, 1)  # (seq_length, batch_size, hidden_size)
        
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None
        
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        
        x = x.transpose(0, 1)  # (batch_size, seq_length, hidden_size)
    
        mean_token = self.dropout(x.mean(dim=1))
        logits = self.classifier(mean_token)  # (batch_size, num_labels)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )