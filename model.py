import torch.nn as nn
import torch

class EmotionRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text input is expected to be [batch_size, sequence_length]
        embedded = self.dropout(self.embedding(text))
        # embedded becomes [batch_size, sequence_length, embedding_dim]

        output, (hidden, cell) = self.lstm(embedded)
        # output is [batch_size, sequence_length, hidden_dim * num_directions]
        # hidden is [num_layers * num_directions, batch_size, hidden_dim]

        # We take the final hidden state of the last layer for classification.
        # For a bidirectional LSTM, the last hidden state contains forward and backward info.
        # hidden[-2,:,:] is the last forward hidden state
        # hidden[-1,:,:] is the last backward hidden state
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden becomes [batch_size, hidden_dim * 2]

        return self.fc(hidden)
