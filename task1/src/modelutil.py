from typing import Optional, Union, Tuple

import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput

from config import labels as LABELS

label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for idx, label in enumerate(LABELS)}


class CustomBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # weight calculation
            num_of_zeros = torch.sum(0 == labels)
            num_of_pos = labels.view(-1, 23).sum(dim=0)
            pos_weights = num_of_zeros / num_of_pos
            pos_weights = pos_weights.detach()
            #
            # loss calculation
            loss_fct = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weights)
            loss = loss_fct(logits, labels.transpose(1, 2).float())
            loss = (loss * attention_mask.unsqueeze(dim=2)).mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def model_init(model_name, pretrain=True):
    if pretrain:
        return CustomBertForTokenClassification.from_pretrained(
            model_name, id2label=id2label, label2id=label2id
        )

    config = AutoConfig.from_pretrained(
        model_name, num_labels=len(LABELS), trust_remote_code=False)
    return CustomBertForTokenClassification(config)
