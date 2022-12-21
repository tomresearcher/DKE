import torch
import torch.nn as nn


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, value_head):
        super().__init__()
        self.value_head = value_head
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if value_head != 0:
            self.denseFeature1 = nn.Linear(value_head, (value_head*100))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size + value_head, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        if self.value_head != 0:
            x1 = kwargs.pop("externalFeatures", None)
            x = torch.cat((x, x1), dim=-1)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
