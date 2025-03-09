"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,5"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
time.sleep(0.1)
from torch import nn, optim
from torch.optim import AdamW
from tqdm import tqdm


from data_test import *
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
model.apply(initialize_weights)
model = DataParallelModel(model)
model.cuda()
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
        trg = batch
        optimizer.zero_grad()
        output = model(trg)
        output_reshape = [out.contiguous().view(-1, out.shape[-1]) for out in output]
        # trg = trg[:, 1:].contiguous().view(-1)
        trg = trg.contiguous().view(-1)
        loss = criterion(output_reshape, trg)

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
            trg = batch
            output = model(trg)
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg.contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                    trg_words = idx_to_word(batch[j], tokenizer.encoder)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, tokenizer.encoder)
                    break

            f = open('transformer/test_result.txt', 'w')
            f.write(str(output_words)+'\n')
            f.write('---------------------------------------------------\n')
            f.write(str(idx_to_word(trg, tokenizer.encoder)))
            f.close()
            break

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # Track the frequency of each token in the generated sequence
    token_counts = torch.zeros(enc_voc_size, device=device)  # Assuming model has config with vocab_size

    for _ in range(max_new_tokens):
        idx_cond = idx
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        # Apply top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # Apply temperature scalingß
        if temperature > 0.0:
            logits = logits / temperature
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1)
            # idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = logits.max(dim=-1,keepdim= True)[1]
        else:
            # idx_next = torch.argmax(logits, dim=1, keepdim=True)
            
            idx_next = logits.max(dim=-1,keepdim= True)[1]
            

        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for i, batch in enumerate(valid_iter):    
        trg = batch[:,:10].to(device)
        token_ids = generate(
            model=model,
            idx=trg,
            max_new_tokens=400,
            context_size=max_len,
            top_k=1,
            temperature=0.0,
            eos_id=tokenizer.encoder.encode("<eos>"
                                ,allowed_special = {"<eos>"})
        )
        # print(idx_to_word(trg[0], tokenizer.encoder))
        # print(token_ids)
        print(idx_to_word(token_ids[0], tokenizer.encoder))
        break
    # evaluate(model, valid_iter, criterion)




if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
