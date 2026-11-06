"""Assignment 2: frozen DINOv2 ViT-S/14, linear probe, ablation, and retrieval.

Run in Colab with internet enabled and preferably an NVIDIA GPU.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from transformers import AutoModel

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
MODEL_ID = 'facebook/dinov2-small'
A1_RESULTS = {
    'simple_cnn': {'accuracy': 0.633, 'macro_f1': 0.584, 'macro_precision': 0.679, 'macro_recall': 0.633},
    'resnet18': {'accuracy': 0.933, 'macro_f1': 0.933, 'macro_precision': 0.933, 'macro_recall': 0.933}
}


class FoodPaths(Dataset):
    def __init__(self, samples, image_transform):
        self.samples = samples
        self.transform = image_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            image = self.transform(im.convert('RGB'))
        return image, int(label)


def get_transform(augmentation=False):
    pipeline = [transforms.Resize((224, 224))]
    if augmentation:
        pipeline += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                     transforms.ColorJitter(brightness=0.2, contrast=0.2)]
    pipeline += [transforms.ToTensor(), transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]
    return transforms.Compose(pipeline)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def embed_dataset(encoder, samples, augmentation, batch_size, device, num_workers=0):
    loader = DataLoader(FoodPaths(samples, get_transform(augmentation)), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    features, labels = [], []
    encoder.eval()
    for images, y in loader:
        images = images.to(device)
        if images.shape[-1] % 14 != 0:
            raise ValueError('DINOv2 patch size is 14: expected dimensions divisible by 14')
        out = encoder(pixel_values=images)
        # CLS token: 384-d for DINOv2-small, used for classification and similarity search.
        x = out.last_hidden_state[:, 0, :]
        features.append(x.cpu())
        labels.append(y)
    return torch.cat(features).float(), torch.cat(labels).long()


def eval_head(head, X, y):
    head.eval()
    with torch.inference_mode():
        logits = head(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        actual = y.cpu().numpy()
        loss = float(nn.functional.cross_entropy(logits, y).item())
        return {'accuracy': float(accuracy_score(actual, preds)),
                'macro_f1': float(f1_score(actual, preds, average='macro', zero_division=0)),
                'loss': loss}, probs, preds


def fit_head(train_X, train_y, val_X, val_y, nclasses, epochs, lr, seed, output, name):
    # Both ablation variants have exactly one feature vector per original training image.
    set_seed(seed)
    head = nn.Linear(train_X.shape[1], nclasses)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.02)
    loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
    best = None
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        metrics, _, _ = eval_head(head, val_X, val_y)
        record = {'epoch': epoch, 'train_loss': total_loss / len(train_y),
                  'val_loss': metrics['loss'], 'val_accuracy': metrics['accuracy'],
                  'val_macro_f1': metrics['macro_f1']}
        history.append(record)
        score = (metrics['macro_f1'], metrics['accuracy'], -metrics['loss'])
        if best is None or score > best['score']:
            best = {'score': score, 'epoch': epoch,
                    'state': {k: v.detach().clone() for k, v in head.state_dict().items()}}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f'{name}: epoch {epoch}/{epochs}, train loss={record["train_loss"]:.4f}, '
                  f'val acc={record["val_accuracy"]:.3f}, macro-F1={record["val_macro_f1"]:.3f}', flush=True)
    head.load_state_dict(best['state'])
    metrics, probabilities, preds = eval_head(head, val_X, val_y)
    metrics['best_epoch'] = best['epoch']
    metrics['training_samples'] = len(train_y)
    with (output / f'history_{name}.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return head, metrics, history, probabilities, preds


def plot_curves(histories, output):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for name, history in histories.items():
        xs = [h['epoch'] for h in history]
        for ax, key, title in zip(axs, ['train_loss','val_accuracy','val_macro_f1'],
                                  ['Training loss','Validation accuracy','Validation macro F1']):
            ax.plot(xs, [h[key] for h in history], label=name)
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png', dpi=160)
    plt.close(fig)


def plot_cm(truth, pred, classes, dest):
    cm = confusion_matrix(truth, pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap='Blues')
    for (i,j), count in np.ndenumerate(cm):
        ax.text(j,i,str(count),ha='center',va='center',color='black')
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='True', title='DINOv2 validation confusion matrix')
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def retrieval_precision_at_k(query_X, query_y, train_X, train_y, k=3):
    qry = nn.functional.normalize(query_X, dim=1)
    train = nn.functional.normalize(train_X, dim=1)
    sims = qry @ train.T
    top = sims.topk(k, dim=1).indices
    precision = (train_y[top] == query_y[:, None]).float().mean(dim=1)
    return float(precision.mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='food_dataset')
    parser.add_argument('--output-dir', default='artifacts')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = parser.parse_args()
    set_seed(args.seed)
    device_name = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no GPU found. Use --device cpu or enable Colab T4.')
    device = torch.device(device_name)
    root = Path(args.data_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_ds = datasets.ImageFolder(root / 'train')
    val_ds = datasets.ImageFolder(root / 'val')
    if train_ds.classes != val_ds.classes:
        raise ValueError(f'Class mismatch: {train_ds.classes} vs {val_ds.classes}')
    classes = train_ds.classes
    print('Classes:', classes, 'train:', len(train_ds), 'val:', len(val_ds), 'device:', device, flush=True)
    if len(classes) != 3:
        raise ValueError('This assignment expects exactly 3 classes')

    print('Loading DINOv2 encoder (first time downloads public checkpoint)...', flush=True)
    encoder = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    print('Extracting original training embeddings...', flush=True)
    train_X, train_y = embed_dataset(encoder, train_ds.samples, False, args.batch_size, device)
    print('Extracting fixed augmented training embeddings...', flush=True)
    set_seed(args.seed + 1)
    augmented_X, augmented_y = embed_dataset(encoder, train_ds.samples, True, args.batch_size, device)
    set_seed(args.seed)
    print('Extracting validation embeddings...', flush=True)
    val_X, val_y = embed_dataset(encoder, val_ds.samples, False, args.batch_size, device)
    if not torch.equal(train_y, augmented_y):
        raise RuntimeError('Augmentation accidentally changed label order')

    histories, variants, heads, probabilities = {}, {}, {}, {}
    for name, X in [('no_augmentation', train_X), ('with_augmentation', augmented_X)]:
        head, metrics, hist, probs, preds = fit_head(X, train_y, val_X, val_y,
                     len(classes), args.epochs, args.lr, args.seed, output, name)
        variants[name] = metrics
        histories[name] = hist
        heads[name] = head
        probabilities[name] = probs
        print(f'{name}: {json.dumps(metrics)}', flush=True)

    # This choice is made using validation metrics; a separate holdout test is recommended.
    selected = sorted(variants, key=lambda n: (variants[n]['macro_f1'], variants[n]['accuracy'], -variants[n]['loss']), reverse=True)[0]
    head = heads[selected]
    selected_metrics, probs, preds = eval_head(head, val_X, val_y)
    report = classification_report(val_y.numpy(), preds, target_names=classes,
                                   output_dict=True, zero_division=0)
    top1 = selected_metrics['accuracy']
    p3 = retrieval_precision_at_k(val_X, val_y, train_X, train_y, k=3)
    plot_curves(histories, output)
    plot_cm(val_y.numpy(), preds, classes, output / 'confusion_matrix.png')
    normalized = nn.functional.normalize(train_X, dim=1).numpy().astype(np.float32)
    paths = np.array([str(Path(p).relative_to(root)) for p, _ in train_ds.samples])
    np.savez_compressed(output / 'train_index.npz', embeddings=normalized,
                        paths=paths, labels=train_y.numpy())
    torch.save({'state_dict': head.state_dict(), 'classes': classes,
                'model_id': MODEL_ID, 'feature_dim': int(train_X.shape[1]),
                'selected_variant': selected}, output / 'classifier.pt')
    with (output / 'val_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['file','true','pred','confidence'] + [f'prob_{c}' for c in classes]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (file, label), pred, prob in zip(val_ds.samples, preds, probs):
            row = {'file': str(Path(file).relative_to(root)), 'true':classes[label],
                   'pred':classes[int(pred)], 'confidence':round(float(prob.max()),5)}
            row.update({f'prob_{c}':round(float(prob[i]),5) for i,c in enumerate(classes)})
            writer.writerow(row)

    summary = {'dataset': {'train_count':len(train_ds), 'val_count':len(val_ds),
                           'classes':classes}, 'seed':args.seed,
               'model':MODEL_ID, 'encoder_trainable':False,
               'head':'Linear 384 to 3', 'epoch_budget':args.epochs,
               'a1_reported_not_retrained': A1_RESULTS,
               'ablation': variants, 'selected_variant': selected,
               'selected': {'top1_accuracy':top1, 'macro_f1':selected_metrics['macro_f1'],
                            'retrieval_precision_at_3':p3},
               'classification_report':report,
               'evaluation_note':'The same 60-image val set is used for selecting the head and reporting its score; no independent test set.',
               'synthetic_data_used':False,
               'augmentation_note':'One fixed augmented view per train photo, compared with one original view; same number of examples and optimizer steps.'}
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nDONE. Selected:', selected, 'Top-1:', round(top1,4),
          'macro F1:', round(selected_metrics['macro_f1'],4), 'P@3:', round(p3,4), flush=True)
    print('Outputs:', output.resolve(), flush=True)


if __name__ == '__main__':
    main()
