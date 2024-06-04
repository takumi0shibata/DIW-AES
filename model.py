import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BERT_Regressor(nn.Module):

    def __init__(
        self,
        model_name: str,
        num_labels:int = 1
    ) -> None:
        
        super(BERT_Regressor, self).__init__()

        config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.regressor(pooled_output)

        return self.sigmoid(logits)