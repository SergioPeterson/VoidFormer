import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics import BLEUScore, CharErrorRate, WordErrorRate
from tqdm import tqdm

from config import get_config, latest_weights_file_path
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from train import get_ds

def load_latest_model(config, device):
    model_path = latest_weights_file_path(config)
    if model_path is None:
        raise FileNotFoundError("No saved model weights found.")

    print(f"Loading model from: {model_path}")
    state = torch.load(model_path, map_location=device)

    model = build_transformer(
        config['vocab_src_len'],
        config['vocab_tgt_len'],
        config['seq_len'],
        config['seq_len'],
        d_model=config['d_model']
    ).to(device)
    model.load_state_dict(state['model_state_dict'])

    return model

def run_evaluation(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the tokenizer and dataset
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Add vocabulary sizes to the config for model construction
    config['vocab_src_len'] = tokenizer_src.get_vocab_size()
    config['vocab_tgt_len'] = tokenizer_tgt.get_vocab_size()

    # Load the latest model
    model = load_latest_model(config, device)
    model.eval()

    # Metrics
    bleu = BLEUScore()
    cer = CharErrorRate()
    wer = WordErrorRate()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            # Decode predictions
            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len=config['seq_len'],
                device=device
            )

            # Convert tokens to text
            target_text = batch['tgt_text'][0]
            predicted_text = tokenizer_tgt.decode(model_out.cpu().numpy())

            all_targets.append(target_text)
            all_predictions.append(predicted_text)

    # Compute metrics
    bleu_score = bleu(all_predictions, all_targets)
    cer_score = cer(all_predictions, all_targets)
    wer_score = wer(all_predictions, all_targets)

    print("Evaluation Results:")
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"Character Error Rate (CER): {cer_score:.4f}")
    print(f"Word Error Rate (WER): {wer_score:.4f}")

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

if __name__ == '__main__':
    config = get_config()
    run_evaluation(config)
