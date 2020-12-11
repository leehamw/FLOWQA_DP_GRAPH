import math
import random
import msgpack
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

# ------------------------------------------------------------------------------
# Neural Modules
# ------------------------------------------------------------------------------

def set_seq_dropout(option): # option = True or False
    global do_seq_dropout
    do_seq_dropout = option

def set_my_dropout_prob(p): # p between 0 to 1
    global my_dropout_p
    my_dropout_p = p

def seq_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training == False or p == 0:
        return x
    dropout_mask = 1.0 / (1-p) * torch.bernoulli((1-p) * (x.new_zeros(x.size(0), x.size(2)) + 1))
    return dropout_mask.unsqueeze(1).expand_as(x) * x

def dropout(x, p=0, training=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if do_seq_dropout and len(x.size()) == 3: # if x is (batch * len * input_size)
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)

class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type=nn.LSTM, concat_layers=False, do_residual=False, add_feat=0, dialog_flow=False, bidir=True):
        super(StackedBRNN, self).__init__()
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.do_residual = do_residual
        self.dialog_flow = dialog_flow
        self.hidden_size = hidden_size

        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else (2 * hidden_size + add_feat if i == 1 else 2 * hidden_size)
            if self.dialog_flow == True:
                input_size += 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,num_layers=1,bidirectional=bidir))

    def forward(self, x, x_mask=None, return_list=False, additional_x=None, previous_hiddens=None):
        # return_list: return a list for layers of hidden vectors
        # Transpose batch and sequence dims
        #
        x = x.transpose(0, 1) ##barch,len_d,? ->len_d,batch,?
        if additional_x is not None:
            additional_x = additional_x.transpose(0, 1)

        # Encode all layers
        hiddens = [x]
        for i in range(self.num_layers):
            rnn_input = hiddens[-1] # len_d,batch,?
            if i == 1 and additional_x is not None:
                rnn_input = torch.cat((rnn_input, additional_x), 2)
            # Apply dropout to input
            if my_dropout_p > 0:
                rnn_input = dropout(rnn_input, p=my_dropout_p, training=self.training)
            if self.dialog_flow == True:
                if previous_hiddens is not None:
                    dialog_memory = previous_hiddens[i-1].transpose(0, 1)
                else:
                    #新建一个与rnn_input类型相同的Variable
                    dialog_memory = rnn_input.new_zeros((rnn_input.size(0), rnn_input.size(1), self.hidden_size * 2)) #len_d, batch, 2*hidden
                rnn_input = torch.cat((rnn_input, dialog_memory), 2)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0] # (seq_len, batch, hidden_size)
            if self.do_residual and i > 0:
                rnn_output = rnn_output + hiddens[-1]
            hiddens.append(rnn_output)

        # Transpose back
        hiddens = [h.transpose(0, 1) for h in hiddens] # (batch, seq_len, hidden_size)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(hiddens[1:], 2)
        else:
            output = hiddens[-1]

        if return_list:
            return output, hiddens[1:]
        else:
            return output

class GraphEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, round=3):
        super(GraphEncoder, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size*2
        self.weightnode=nn.Parameter(torch.Tensor(input_size,self.hidden_size))
        # self.weighthidden=nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.a=nn.Parameter(torch.Tensor(self.hidden_size))
        torch.nn.init.xavier_normal_(self.weightnode)

        self.scoring2 = AttentionScore(self.hidden_size, self.hidden_size)

        self.update3 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.update4 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))

        torch.nn.init.xavier_normal_(self.update3)
        torch.nn.init.xavier_normal_(self.update4)
        # torch.nn.init.xavier_normal_(self.updatea)
        self.scoring1 = AttentionScore(self.hidden_size, self.hidden_size)

        self.scoring = AttentionScore(self.hidden_size,self.hidden_size)
    def forward(self,doc_hiddens,doc_hiddens_mask, node_emb_expand, node_emb_expand_mask, edge_expand):
        '''

        :param doc_hiddens: #   bsz', len_d, hidden_state_dim
              doc_hiddens_mask#   bsz', len_d
        :param node_emb_expand:   bsz', max_node_num, max_node_length, emb_size
        :param node_emb_expand_mask:  bsz',  max_node_num, max_node_length
        :param  edge_expan:      bsz',  max_node_num, max_node_num
        :return:
        '''


        use_doc_mask=torch.eq(doc_hiddens_mask,0)

        actual_lens=torch.sum(use_doc_mask, dim=1)-1 #bas'
        bsz=doc_hiddens.size(0)
        tmp = []
        for i in range(bsz):
            a = doc_hiddens[i].index_select(dim=0, index=actual_lens[i])
            tmp.append(a.squeeze())
        last_doc_hidden=torch.stack(tmp,dim=0) #[bsz', hidden_state_dim]

        #bsz', max_node_num, max_node_length, emb_size
        use_doc_hidden=last_doc_hidden.unsqueeze(1).expand(last_doc_hidden.size(0),node_emb_expand.size(1),last_doc_hidden.size(-1)).contiguous() #bsz',max_node_num,1, hidden_state_dim

        use_doc_hidden=use_doc_hidden.view(-1,  use_doc_hidden.size(-1))#(bsz'*max_node_num),1, hidden_state_dim
        use_doc_hidden=use_doc_hidden.unsqueeze(1).contiguous()

        #(bsz'), max_node_num，hidden_dim
        # attn_node=node_emb_expand # bsz', max_node_num, max_node_length, emb_size
        attn_node = torch.matmul(node_emb_expand, self.weightnode) #36, 210, 11, 250
        fi_attn_node=attn_node.contiguous().view(-1, attn_node.size(2), attn_node.size(3))  #(bsz'*max_node_num), max_node_length, hidden_size
        # print(fi_attn_node.size()) #7560, 11, 250
        # print('fi_attn_node{}'.format(fi_attn_node))
        score2 = self.scoring2(use_doc_hidden , fi_attn_node ).contiguous()  # #(bsz'*max_node_num)', 1, max_node_length
        e= score2.squeeze() # 7560 11 (bsz'*max_node_num)', 1, max_node_length ->(bsz'*max_node_num)',  max_node_length  hidden (bsz'*max_node_num),max_node_length, hidden_size->(bsz'*max_node_num),max_node_length
        fi_node_emb_expand_mask=node_emb_expand_mask.view(-1,node_emb_expand_mask.size(2)) #7560,11 (bsz'**max_node_num) , max_node_length
        e.data.masked_fill_(fi_node_emb_expand_mask.data, -float('inf'))
        e=F.softmax(e, dim=1) #(bsz'*max_node_num),max_node_length

        #fi_attn_node (bsz'*max_node_num), max_node_length, hidden_size  e (bsz'*max_node_num),max_node_length
        e_u=e.unsqueeze(1)#  7560,1,11 bsz'*max_node_num),1, max_node_length  *fi_attn_node #(bsz'*max_node_num), max_node_length, hidden_size

        node_initial=e_u.bmm(fi_attn_node).squeeze().contiguous()# 7560 1 250 bsz'*max_node_num),1,hidden_size 7560 250

        node_initial= node_initial.view(node_emb_expand.size(0),node_emb_expand.size(1),use_doc_hidden.size(-1)) #bsz', max_node_num',  hidden_size

        # node_initial=fi_initial.view(node_emb_expand.size(0),node_emb_expand.size(1),fi_initial.size(1)) #bsz', max_node_num, hidden
        for i in range(3):
            compu_node_initial=node_initial.contiguous()  #36, 210, 250  bsz', max_node_num, hidden
            score1= self.scoring1(node_initial, compu_node_initial) # #36, 210,210  bsz', max_node_num, max_node_num

        #edge_expan:  bsz', max_node_num, max_node_num
            edge_mask=torch.eq(edge_expand,0)
            score1.data.masked_fill_(edge_mask.data, -100000)
            ea=F.softmax(score1, dim=2) ##36, 210,210 bsz', max_node_num. max_node_num
            fiii=ea.bmm(compu_node_initial)#bsz', max_node_num, hidden 36, 210,250
        # fiii=weight_ea*w_compu_node_initial  #bsz', max_node_num, max_node_num, hidden
        # w_fii=fiii.view(-1, fiii.size(2),fiii.size(3))#(bsz'*max_node_num), max_node_num, hidden
        # #fi_initial (bsz'*max_node_num),hidden
        # out,h0= self.ggru(w_fii, fi_initial.unsqueeze(0)) #(bsz'*max_node_num), max_node_num, hidden  h0 1,(bsz'*max_node_num),hidden
        # #out (bsz'*max_node_num), max_node_num, hidden  output  bsz', max_node_num, max_node_num, hidden
        # output=out.view(node_initial.size(0),fiii.size(1),out.size(1),out.size(-1))
        # print('fiii{}'.format(fiii)) #有问题
        # output=torch.matmul( node_initial, self.update1)+torch.matmul(fiii, self.update2)
            node_initial=fiii
            output = fiii

        #doc_hiddens: #   bsz', len_d ,hidden_state_dim [36, 583, 250]  output  # 36, 210,250
        score=self.scoring(doc_hiddens,output) #bsz', len_d, max_node_num
        tmp_mask=torch.eq(torch.sum(edge_expand,dim=2),0) #bsz,max_node_num
        final_mask=tmp_mask.unsqueeze(1).expand(tmp_mask.size(0),doc_hiddens.size(1),tmp_mask.size(1))
        score.data.masked_fill_(final_mask.data, -float('inf'))
        alpha = F.softmax(score, dim=2) #bsz,lend,node_num
        # print('alpha{}'.format(alpha))
        matched_seq = alpha.bmm(output) #bsz,lend,hidden 36, 583, 250

        matched_seq=torch.matmul(matched_seq, self.update3)+torch.matmul(doc_hiddens, self.update4)
        # print('matched_seq{}'.format(matched_seq))
        return  matched_seq #bsz', len_d, hidden_state_dim











def RNN_from_opt(input_size_, hidden_size_, opt, num_layers=-1, concat_rnn=None, add_feat=0, dialog_flow=False):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    new_rnn = StackedBRNN(
        input_size=input_size_,
        hidden_size=hidden_size_,
        num_layers=num_layers if num_layers > 0 else opt['rnn_layers'],
        rnn_type=RNN_TYPES[opt['rnn_type']],
        concat_layers=concat_rnn if concat_rnn is not None else opt['concat_rnn'],
        do_residual=opt['do_residual_rnn'] or opt['do_residual_everything'],
        add_feat=add_feat,
        dialog_flow=dialog_flow
    )
    output_size = 2 * hidden_size_
    if (concat_rnn if concat_rnn is not None else opt['concat_rnn']):
        output_size *= num_layers if num_layers > 0 else opt['rnn_layers']
    return new_rnn, output_size

class MemoryLasagna_Time(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='lstm'):
        super(MemoryLasagna_Time, self).__init__()
        RNN_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell}

        self.rnn = RNN_TYPES[rnn_type](input_size, hidden_size)
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x, memory):
        if self.training:
            x = x * self.dropout_mask

        memory = self.rnn(x.contiguous().view(-1, x.size(-1)), memory)
        if self.rnn_type == 'lstm':
            h = memory[0].view(x.size(0), x.size(1), -1)
        else:
            h = memory.view(x.size(0), x.size(1), -1)
        return h, memory

    def get_init(self, sample_tensor):
        global my_dropout_p
        self.dropout_mask = 1.0 / (1-my_dropout_p) * torch.bernoulli((1-my_dropout_p) * (sample_tensor.new_zeros(sample_tensor.size(0), sample_tensor.size(1), self.input_size) + 1))

        h = sample_tensor.new_zeros(sample_tensor.size(0), sample_tensor.size(1), self.hidden_size).float()
        memory = sample_tensor.new_zeros(sample_tensor.size(0) * sample_tensor.size(1), self.hidden_size).float()
        if self.rnn_type == 'lstm':
            memory = (memory, memory)
        return h, memory

class MTLSTM(nn.Module):
    def __init__(self, opt, embedding=None, padding_idx=0):
        """Initialize an MTLSTM

        Arguments:
            embedding (Float Tensor): If not None, initialize embedding matrix with specified embedding vectors
        """
        super(MTLSTM, self).__init__()

        self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_dim'], padding_idx=padding_idx)
        if embedding is not None:
            self.embedding.weight.data = embedding

        state_dict = torch.load(opt['MTLSTM_path'])
        self.rnn1 = nn.LSTM(300, 300, num_layers=1, bidirectional=True)
        self.rnn2 = nn.LSTM(600, 300, num_layers=1, bidirectional=True)

        state_dict1 = dict([(name, param.data) if isinstance(param, Parameter) else (name, param)
                        for name, param in state_dict.items() if '0' in name])
        state_dict2 = dict([(name.replace('1', '0'), param.data) if isinstance(param, Parameter) else (name.replace('1', '0'), param)
                        for name, param in state_dict.items() if '1' in name])
        self.rnn1.load_state_dict(state_dict1)
        self.rnn2.load_state_dict(state_dict2)

        for p in self.embedding.parameters():
            p.requires_grad = False
        for p in self.rnn1.parameters():
            p.requires_grad = False
        for p in self.rnn2.parameters():
            p.requires_grad = False

        self.output_size = 600

    def setup_eval_embed(self, eval_embed, padding_idx=0):
        """Allow evaluation vocabulary size to be greater than training vocabulary size

        Arguments:
            eval_embed (Float Tensor): Initialize eval_embed to be the specified embedding vectors
        """
        self.eval_embed = nn.Embedding(eval_embed.size(0), eval_embed.size(1), padding_idx = padding_idx)
        self.eval_embed.weight.data = eval_embed

        for p in self.eval_embed.parameters():
            p.requires_grad = False

    def forward(self, x_idx, x_mask):
        """A pretrained MT-LSTM (McCann et. al. 2017).
        This LSTM was trained with 300d 840B GloVe on the WMT 2017 machine translation dataset.

        Arguments:
            x_idx (Long Tensor): a Long Tensor of size (batch * len).
            x_mask (Byte Tensor): a Byte Tensor of mask for the input tensor (batch * len).
        """
        emb = self.embedding if self.training else self.eval_embed
        x_hiddens = emb(x_idx)

        lengths = x_mask.data.eq(0).long().sum(dim=1) # (batch)
        lens, indices = torch.sort(lengths, 0, True)

        output1, _ = self.rnn1(pack(x_hiddens[indices], lens.tolist(), batch_first=True))
        output2, _ = self.rnn2(output1)

        output1 = unpack(output1, batch_first=True)[0]  #也可以用index_select
        output2 = unpack(output2, batch_first=True)[0]

        _, _indices = torch.sort(indices, 0)
        output1 = output1[_indices]
        output2 = output2[_indices]

        return output1, output2

# Attention layers
class AttentionScore(nn.Module):
    """
    sij = Relu(Wx1)DRelu(Wx2)
    """
    def __init__(self, input_size, attention_hidden_size, similarity_score = False):
        super(AttentionScore, self).__init__()
        self.linear = nn.Linear(input_size, attention_hidden_size, bias=False)

        if similarity_score:
            self.linear_final = Parameter(torch.ones(1, 1, 1) / (attention_hidden_size ** 0.5), requires_grad = False)
        else:
            self.linear_final = Parameter(torch.ones(1, 1, attention_hidden_size), requires_grad = True)

    def forward(self, x1, x2):
        """
        x1: batch * len1 * input_size
        x2: batch * len2 * input_size
        scores: batch * len1 * len2 <the scores are not masked>
        """
        x1 = dropout(x1, p=my_dropout_p, training=self.training)
        x2 = dropout(x2, p=my_dropout_p, training=self.training)

        x1_rep = self.linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), x1.size(1), -1)
        x2_rep = self.linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), x2.size(1), -1)

        x1_rep = F.relu(x1_rep)
        x2_rep = F.relu(x2_rep)
        final_v = self.linear_final.expand_as(x2_rep) #batch * len2 * hidden_size

        x2_rep_v = final_v * x2_rep
        scores = x1_rep.bmm(x2_rep_v.transpose(1, 2))
        return scores


class GetAttentionHiddens(nn.Module):
    def __init__(self, input_size, attention_hidden_size, similarity_attention = False):
        super(GetAttentionHiddens, self).__init__()
        self.scoring = AttentionScore(input_size, attention_hidden_size, similarity_score=similarity_attention)

    def forward(self, x1, x2, x2_mask, x3=None, scores=None, return_scores=False, drop_diagonal=False):
        """
        Using x1, x2 to calculate attention score, but x1 will take back info from x3.
        If x3 is not specified, x1 will attend on x2.

        x1: batch * len1 * x1_input_size
        x2: batch * len2 * x2_input_size
        x2_mask: batch * len2

        x3: batch * len2 * x3_input_size (or None)
        """
        if x3 is None:
            x3 = x2

        if scores is None:
            scores = self.scoring(x1, x2) #batch, lend, lenq

        # Mask padding
        x2_mask = x2_mask.unsqueeze(1).expand_as(scores)
        scores.data.masked_fill_(x2_mask.data, -float('inf'))
        if drop_diagonal:
            assert(scores.size(1) == scores.size(2))
            diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)
            scores.data.masked_fill_(diag_mask, -float('inf')) #在mask值为1的位置处用value填充

        # Normalize with softmax
        alpha = F.softmax(scores, dim=2) # batch * len1 * len2

        # Take weighted average
        matched_seq = alpha.bmm(x3) # batch * len1 * x2_input_size
        if return_scores:
            return matched_seq, scores
        else:
            return matched_seq

class DeepAttention(nn.Module):
    def __init__(self, opt, abstr_list_cnt, deep_att_hidden_size_per_abstr, do_similarity=False, word_hidden_size=None, do_self_attn=False, dialog_flow=False, no_rnn=False):
        super(DeepAttention, self).__init__()

        self.no_rnn = no_rnn

        word_hidden_size = opt['embedding_dim'] if word_hidden_size is None else word_hidden_size
        abstr_hidden_size = opt['hidden_size'] * 2

        att_size = abstr_hidden_size * abstr_list_cnt + word_hidden_size

        self.int_attn_list = nn.ModuleList()
        for i in range(abstr_list_cnt+1):
            self.int_attn_list.append(GetAttentionHiddens(att_size, deep_att_hidden_size_per_abstr, similarity_attention=do_similarity))

        rnn_input_size = abstr_hidden_size * abstr_list_cnt * 2 + (opt['hidden_size'] * 2)

        self.att_final_size = rnn_input_size
        if not self.no_rnn:
            self.rnn, self.output_size = RNN_from_opt(rnn_input_size, opt['hidden_size'], opt, num_layers=1, dialog_flow=dialog_flow)
        #print('Deep attention x {}: Each with {} rays in {}-dim space'.format(abstr_list_cnt, deep_att_hidden_size_per_abstr, att_size))
        #print('Deep attention RNN input {} -> output {}'.format(self.rnn_input_size, self.output_size))

        self.opt = opt
        self.do_self_attn = do_self_attn

    def forward(self, x1_word, x1_abstr, x2_word, x2_abstr, x1_mask, x2_mask, return_bef_rnn=False, previous_hiddens=None):
        """
        x1_word: batch*q_num),len_d,2*emb_size   torch.cat([x1_emb_expand, x1_cove_high_expand], 2)]
        x1_abstr:[(batch*q_num, len_d, hidden_size),(batch*q_num, len_d, hidden_size)]  doc_abstr_ls
        x2_word: (batch*q_num),len_q,2*embsize)  [torch.cat([x2_emb, x2_cove_high], 2)]
        x2_abstr:[(batch*q_num), len_q, hidden_size),(batch*q_num), len_q, hidden_size)]
        x1_word, x2_word, x1_abstr, x2_abstr are list of 3D tensors.
        x1_mask:(batch*q_num, len_d)
        x2_mask:(batch*q_num, len_q)
        3D tensor: batch_size * length * hidden_size
        """
        # the last tensor of x2_abstr is an addtional tensor
        x1_att = torch.cat(x1_word + x1_abstr, 2)  #batch*q_num),len_d,4*emb_size
        x2_att = torch.cat(x2_word + x2_abstr[:-1], 2) #
        x1 = torch.cat(x1_abstr, 2) #batch*q_num),len_d,2*emb_size

        x2_list = x2_abstr
        for i in range(len(x2_list)):
            attn_hiddens = self.int_attn_list[i](x1_att, x2_att, x2_mask, x3=x2_list[i], drop_diagonal=self.do_self_attn) # batch*q_num * len1 * x2_input_size
            x1 = torch.cat((x1, attn_hiddens), 2)

        if not self.no_rnn:
            x1_hiddens = self.rnn(x1, x1_mask, previous_hiddens=previous_hiddens)
            if return_bef_rnn:
                return x1_hiddens, x1
            else:
                return x1_hiddens
        else:
            return x1

# For summarizing a set of vectors into a single vector
class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x = dropout(x, p=my_dropout_p, training=self.training)

        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)
        return alpha

# For attending the span in document from the query
class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = x_i'Wy for x_i in X.
    """
    def __init__(self, x_size, y_size, opt, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = dropout(x, p=my_dropout_p, training=self.training)
        y = dropout(y, p=my_dropout_p, training=self.training)

        Wy = self.linear(y) if self.linear is not None else y  #batch * h1 * 1
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2) #batch * len
        xWy.data.masked_fill_(x_mask.data, -float('inf')) #把1的位置替换成-float('inf')
        return xWy

class GetSpanStartEnd(nn.Module):
    # supports MLP attention and GRU for pointer network updating
    def __init__(self, x_size, h_size, opt, do_indep_attn=True, attn_type="Bilinear", do_ptr_update=True):
        super(GetSpanStartEnd, self).__init__()

        self.attn  = BilinearSeqAttn(x_size, h_size, opt)
        self.attn2 = BilinearSeqAttn(x_size, h_size, opt) if do_indep_attn else None

        self.rnn = nn.GRUCell(x_size, h_size) if do_ptr_update else None

    def forward(self, x, h0, x_mask):
        """
        x = batch * len * x_size
        h0 = batch * h_size
        x_mask = batch * len
        """
        st_scores = self.attn(x, h0, x_mask) # batch * len
        # st_scores = batch * len

        if self.rnn is not None:
            #batch * 1 * len   batch * len * x_size  batch * 1 * x_size   batch *  x_size
            ptr_net_in = torch.bmm(F.softmax(st_scores, dim=1).unsqueeze(1), x).squeeze(1)
            ptr_net_in = dropout(ptr_net_in, p=my_dropout_p, training=self.training)
            h0 = dropout(h0, p=my_dropout_p, training=self.training)
            h1 = self.rnn(ptr_net_in, h0)
            # h1 same size as h0
        else:
            h1 = h0

        end_scores = self.attn(x, h1, x_mask) if self.attn2 is None else\
                     self.attn2(x, h1, x_mask)
        # end_scores = batch * len
        return st_scores, end_scores

class BilinearLayer(nn.Module):
    def __init__(self, x_size, y_size, class_num):
        super(BilinearLayer, self).__init__()
        self.linear = nn.Linear(y_size, x_size * class_num)
        self.class_num = class_num

    def forward(self, x, y):
        """
        x = batch * h1
        y = batch * h2
        """
        x = dropout(x, p=my_dropout_p, training=self.training)
        y = dropout(y, p=my_dropout_p, training=self.training)

        Wy = self.linear(y) #batch , class_num, h1
        Wy = Wy.view(Wy.size(0), self.class_num, x.size(1))
        xWy = torch.sum(x.unsqueeze(1).expand_as(Wy) * Wy, dim=2)
        return xWy.squeeze(-1) # size = batch * class_num

# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------

# by default in PyTorch, +-*/ are all element-wise
def uniform_weights(x, x_mask): # used in lego_reader.py
    """Return uniform weights over non-masked input."""
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha

# bmm: batch matrix multiplication
# unsqueeze: add singleton dimension
# squeeze: remove singleton dimension
def weighted_avg(x, weights): # used in lego_reader.py
    """ x = batch * len * d
        weights = batch * 1* len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)
