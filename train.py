"""
https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html?highlight=cbow
"""
import pickle
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from lib.data_process import save_wordvec_dict, make_context_vector
from lib.model import CBOW
from lib.dataset import WordDataset
from lib.config import Config


def train(config, model, dataloader):
    optimizer = optim.SGD(model.parameters(), config.lr)
    loss_function = nn.NLLLoss()

    for epo in trange(config.epoch):
        total_loss, i = 0, 0
        for context, target in tqdm(dataloader):
            model.zero_grad()
            train_predict = model(context).to(config.device)
            loss = loss_function(train_predict, target)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print("batch loss = ", loss.item())
            total_loss += loss.item()
            i += 1
        print("epo losses = ", total_loss / len(dataloader))

    save_to = os.path.join(config.output_dir, config.model_file)
    torch.save(model.state_dict(), save_to)
    print(f">>> model saved in {save_to}")
    return model


def predict_word(model, context, word_to_idx, idx_to_word, device):
    context_vector = make_context_vector(context, word_to_idx, device)
    predict = model(context_vector).data.cpu().numpy()
    max_idx = np.argmax(predict)
    print('Prediction: {}'.format(idx_to_word[max_idx]))


if __name__ == '__main__':
    torch.manual_seed(1)

    word_dataset = WordDataset(Config)
    vocab_size = word_dataset.vocab_size
    word_to_idx = word_dataset.get_word2idx()

    cbow = CBOW(vocab_size, Config.embedding_dim).to(Config.device)
    model = train(Config, cbow, word_dataset)

    W = model.embeddings.weight.cpu().detach().numpy()
    save_to = os.path.join(Config.output_dir, Config.word_to_vec_file)
    with open(save_to, 'wb') as f:
        pickle.dump(W, f)
    print(f">>> word_vector saved in {save_to}")

    save_wordvec_dict(Config, W, word_to_idx)

