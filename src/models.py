"""Custom models for few-shot learning specific operations."""

import torch
import torch.nn as nn
import transformers
from transformers.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead, RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput
import numpy as np
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))



        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output



class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

    def contrastive_loss(self, sentence_embedding, label):
        batch_num = len(sentence_embedding)
        criterion = nn.CrossEntropyLoss()
        cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        loss = 0
        label = label.cpu()
        label = label.numpy()
        for i in range(batch_num):
            for j in range(batch_num):
                sim = cos_sim(sentence_embedding[i], sentence_embedding[j])
                # logit_sim = torch.tensor([(1 - sim) * 50, (1 + sim) * 50])
                sim = sim.unsqueeze(0)
                logit_sim = torch.cat(((1 - sim) * 50, (1 + sim) * 50),dim=-1)
                if label[i] == label[j]:
                    loss += criterion(logit_sim.view(-1, logit_sim.size(-1)), (torch.tensor(1, device='cuda:0').view(-1)))
                else:
                    loss += criterion(logit_sim.view(-1, logit_sim.size(-1)), (torch.tensor(0, device='cuda:0').view(-1)))
        loss = loss / (batch_num * batch_num - batch_num)
        loss = loss / 100
        return loss

    def contrastive_loss2(self, sentence_embedding, label):
        T = 0.5  
        label = label
        n = label.shape[0]  # batch
        representations = sentence_embedding
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        mask = torch.ones_like(similarity_matrix, device='cuda:0', requires_grad=True) * (label.expand(n, n).eq(label.expand(n, n).t()))
        mask_no_sim = torch.ones_like(mask, device='cuda:0', requires_grad=True) - mask
        mask_dui_jiao_0 = torch.ones(n ,n, device='cuda:0', requires_grad=True) - torch.eye(n, n , device='cuda:0', requires_grad=True)
        
        similarity_matrix = torch.exp(similarity_matrix/T)
        similarity_matrix = similarity_matrix*mask_dui_jiao_0
        sim = mask*similarity_matrix
        no_sim = similarity_matrix - sim
        no_sim_sum = torch.sum(no_sim , dim=1)
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum  = sim + no_sim_sum_expend
        loss = torch.div(sim , sim_sum)
        loss = mask_no_sim + loss + torch.eye(n, n ,device='cuda:0', requires_grad=True)
        loss = -torch.log(loss)  #求-log
        loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n
        return loss
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        alpha = 0
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)
        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        con_loss = self.contrastive_loss(prediction_mask_scores, labels)
        loss = loss + con_loss * alpha
        return ((loss,) + output) if loss is not None else output


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def contrastive_loss(self, sentence_embedding, label):
        batch_num = len(sentence_embedding)
        criterion = nn.CrossEntropyLoss()
        cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        loss = 0
        label = label.cpu()
        label = label.numpy()
        for i in range(batch_num):
            for j in range(batch_num):
                sim = cos_sim(sentence_embedding[i], sentence_embedding[j])
                # logit_sim = torch.tensor([(1 - sim) * 50, (1 + sim) * 50])
                sim = sim.unsqueeze(0)
                logit_sim = torch.cat(((1 - sim) * 50, (1 + sim) * 50),dim=-1)
                if label[i] == label[j]:
                    loss += criterion(logit_sim.view(-1, logit_sim.size(-1)), (torch.tensor(1, device='cuda:0').view(-1)))
                else:
                    loss += criterion(logit_sim.view(-1, logit_sim.size(-1)), (torch.tensor(0, device='cuda:0').view(-1)))
        loss = loss / (batch_num * batch_num - batch_num)
        loss = loss / 100
        return loss

    def contrastive_loss2(self, sentence_embedding, label):
        T = 0.3
        label = label
        n = label.shape[0]  # batch
        representations = sentence_embedding
        matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        # matrix = torch.mm(sentence_embedding, sentence_embedding.t())
        # label_mask = torch.zeros(n ,n, device='cuda:0', requires_grad=True)
        
        label_number_mask = torch.zeros(n ,n, device='cuda:0')

        np_label = label.cpu()
        np_label = np_label.numpy()
        label_dic = {}
        for i in range(n):
            if label_dic.get(np_label[i]) is not None :
                label_dic[np_label[i]] += 1
            else:
                label_dic[np_label[i]] = 1
        for i in range(n):
            tmp = label_dic[np_label[i]]
            if tmp > 1:
                for j in range(n):
                    label_number_mask[i][j] = -(1/(tmp - 1 ))
        
        label_mask = torch.ones_like(matrix, device='cuda:0', requires_grad=True) * (label.expand(n, n).eq(label.expand(n, n).t()))
        label_mask = label_mask * label_number_mask

        mask_dui_jiao_0 = torch.ones(n ,n, device='cuda:0', requires_grad=True) - torch.eye(n, n , device='cuda:0', requires_grad=True)

        matrix = torch.exp(matrix/T)

        matrix_1 = matrix * mask_dui_jiao_0

        matrix_sum = torch.sum(matrix_1 , dim=1)
        # logger.info(matrix_sum)

        loss = torch.div(matrix, matrix_sum)

        loss = torch.log(loss)
        loss = loss * mask_dui_jiao_0
        loss = loss * label_mask
        loss = torch.sum(torch.sum(loss, dim=1) ) #将所有数据都加起来

        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        alpha = 0,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
        
        # logger.info(sequence_output[:, 0, :])
        # logger.info("#############")
        logits = self.classifier(sequence_output)
        # logger.info(logits)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        con_loss = self.contrastive_loss2(sequence_output[:, 0, :], labels)

        loss = (1 - alpha) * loss + con_loss * alpha

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )