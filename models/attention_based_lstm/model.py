import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# vocabulary_size = 8268
# embedding_dim = 128
# aspect_dim = 128

vocabulary_size = 10
embedding_dim = 6  # d
aspect_dim = 3  # da


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.word2vec = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)

        self.weight_for_hidden = nn.Parameter(
            torch.Tensor(embedding_dim, embedding_dim).normal_(mean=0.0, std=1))  # shape(d, d)

        self.weight_for_aspect = nn.Parameter(
            torch.Tensor(aspect_dim, aspect_dim).normal_(mean=0.0, std=1))  # shape(da, da)

        self.weight_for_M = nn.Parameter(
            torch.Tensor(embedding_dim + aspect_dim, 1).normal_(mean=0.0, std=1))  # shape(1, d, da)

        self.linear_for_R = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linear_for_H = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.linear_for_S = nn.Linear(in_features=embedding_dim, out_features=4)

    def forward(self, sentences, lengths, aspect):
        """

        :param sentences: shape(b * max_length)
        :param lengths:  shape(b,)
        :param aspect: shape(da,)
        :return:
        """
        sentences = self.word2vec(sentences)  # b * max_length * d
        aspect = aspect.unsqueeze(0)

        pack_sequence = pack_padded_sequence(sentences, lengths=lengths, batch_first=True)
        pack_hs, (h_n, _) = self.lstm(pack_sequence)
        h_n = h_n.squeeze()
        # hs: shape(b, max_length, d); b_n : shape(b, d)
        hs, lengths = pad_packed_sequence(pack_hs, batch_first=True)

        # shape(b, max_length, d) * shape(d, d) => shape(b, max_length,d)
        weighted_hs = hs.matmul(self.weight_for_hidden)
        weighted_aspect = aspect.matmul(self.weight_for_aspect)  # shape(1, da) * shape(da, da) => shape(1, da)

        weighted_aspect_copies_array = []
        for length in lengths:
            # shape(max_length, da) (结构和 hs 相同，用 0 补齐)
            weighted_aspect_copy = weighted_aspect.mul(torch.cat(
                [torch.ones((length, aspect_dim)), torch.zeros((lengths[0] - length, aspect_dim))]
            ))
            weighted_aspect_copies_array.append(weighted_aspect_copy.unsqueeze(dim=0))

        # shape(b, max_length, da) (结构和 hs 相同，用 0 补齐)
        weighted_aspect_copies = torch.cat(weighted_aspect_copies_array)

        M = torch.cat([weighted_hs, weighted_aspect_copies], dim=2)  # shape(b, max_length, d + da)

        # shape(b, max_length, d + da) * shape(d + da, 1) => shape(b, max_length, 1) => shape(b, 1, max_length)
        a__2 = M.matmul(self.weight_for_M)
        a_1 = a__2.transpose(2, 1)

        alpha = F.softmax(a_1, dim=2)

        # shape(b, 1, max_length) * shape(b, max_length, d) => shape(b, 1, d) => shape(b, d)
        r = alpha.matmul(hs).squeeze()
        h_ = torch.tanh(self.linear_for_R(r) + self.linear_for_H(h_n))  # shape(b, d)
        y = F.softmax(self.linear_for_S(h_), dim=1)  # shape(b, 4)

        return y


## 测试模型使用
if __name__ == '__main__':
    model = Model()

    model(torch.Tensor([
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 0],
        [3, 3, 3, 0, 0],
        [4, 4, 0, 0, 0]]).long(), torch.Tensor([5, 4, 3, 2]).long(), torch.Tensor([3, 3, 3]))
