"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,5"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,6,7"
time.sleep(0.1)
from torch import nn, optim
from torch.optim import AdamW
from tqdm import tqdm


from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from util.parallel import DataParallelModel, DataParallelCriterion



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device,
                    method = method)

print(f'The model has {count_parameters(model):,} trainable parameters')
# model.apply(initialize_weights)
model = DataParallelModel(model)
model.cuda()
model.load_state_dict(torch.load(model_path))
optimizer = AdamW(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
criterion = DataParallelCriterion(criterion)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        for k in range(0,max_len, stride):
            trg = batch[:,k+1:stride+k+1]
            src = batch[:,k:trg.shape[-1]+k]
            optimizer.zero_grad()
            output = model(src)
            output_reshape = [out.contiguous().view(-1, out.shape[-1]) for out in output]
            # trg = trg[:, 1:].contiguous().view(-1)
            trg = trg.contiguous().view(-1)
            loss = criterion(output_reshape, trg)
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()
            print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            src = batch[0]
            trg = batch[1]
            output = model(src)
            output_reshape = [out.contiguous().view(-1, out.shape[-1]) for out in output]
            trg = trg.contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                    trg_words = idx_to_word(batch[j], tokenizer.encoder)
                    k_temp = batch_size//num_worker
                    for k in range(num_worker):
                        output_words = output[k][j//k_temp].max(dim=1)[1]
                        output_words = idx_to_word(output_words, tokenizer.encoder)
                        bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                        total_bleu.append(bleu)

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)
    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, clip)
        # valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(train_loss)

        train_losses.append(train_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'transformer/saved/model-{0}-{1}.pt'.format(train_loss,method))

        f = open(f'transformer/result/train_loss_{method}.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open(f'transformer/result/test_time_{method}.txt', 'w')
        f.write(f'[Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s]')
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
