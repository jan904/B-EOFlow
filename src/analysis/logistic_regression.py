import torch.nn as nn
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from IPython.display import display
from torch.utils.data import DataLoader

from src.model.data_utils import (
    create_latent_adata,
    prepare_train_test_data,
    prepare_data,
    AdataDataset,
)


class LogisticRegressionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        if hidden_dim is None:
            # True logistic regression
            self.model = nn.Linear(input_dim, num_classes)
        else:
            # Small MLP
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes)
            )

    def forward(self, x):
        return self.model(x)  # CrossEntropyLoss applies softmax internally


def train_classifier(
    classifier,
    train_loader,
    val_loader,
    weights,
    device,
    dtype=torch.float32,
    num_epochs=20,
    learning_rate=1e-3,
):
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    best_val_accuracy = 0
    accuracies = []
    train_loss = []
    val_loss = []

    for epoch in tqdm(range(num_epochs)):
        classifier.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch[0].to(device).argmax(dim=1)

            optimizer.zero_grad()
            outputs = classifier(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_loss.append(avg_loss)

        # Validation
        classifier.eval()
        correct, total = 0, 0
        all_predicted = []
        all_labels = []
        with torch.no_grad():
            total_loss = 0
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val[0].to(device).argmax(dim=1)

                outputs = classifier(X_val)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, y_val)
                total_loss += loss.item()
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
                all_predicted.append(predicted.cpu())
                all_labels.append(y_val.cpu())

        accuracy = correct / total
        accuracies.append(accuracy)
        val_loss.append(total_loss / len(val_loader))

        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            all_predicted = torch.cat(all_predicted)
            all_labels = torch.cat(all_labels)
            num_classes = all_labels.max().item() + 1
            class_correct = torch.zeros(num_classes)
            class_total = torch.zeros(num_classes)
            for c in range(num_classes):
                mask = all_labels == c
                class_correct[c] = (all_predicted[mask] == c).sum().item()
                class_total[c] = mask.sum().item()
            best_accuracy_class = (class_correct / class_total.clamp(min=1)).numpy()

    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    return accuracies, train_loss, val_loss, best_accuracy_class


def generate_counterfactuals(analyzer, key):
    dataset, dataloader = prepare_data(
        analyzer.adata,
        batchsize=1024,
        device=analyzer.device,
        dtype=analyzer.dtype,
        label_key=analyzer.conditions,
        shuffle=False,
    )

    cats = dataset.cats[0].categories.tolist()
    key_idx = cats.index(key)

    means = analyzer.model.means.detach()
    key_mean = means[key_idx].to(device=analyzer.device, dtype=analyzer.dtype)

    all_x_cf = []
    all_z_cf = []
    all_source_labels = []

    with torch.no_grad():
        for x_batch, c_batch in dataloader:
            x_batch = x_batch.to(device=analyzer.device, dtype=analyzer.dtype)
            c_batch = [cond.to(device=analyzer.device, dtype=analyzer.dtype) for cond in c_batch]

            z_batch, _ = analyzer.model(x_batch, rev=False)

            # source label per cell
            source_idx = torch.argmax(c_batch[0], dim=1)  # (B,)
            keep_mask = source_idx != key_idx
            if not keep_mask.any():
                continue

            z_batch = z_batch[keep_mask]
            source_idx = source_idx[keep_mask]

            source_labels = [cats[i] for i in source_idx.cpu().tolist()]

            current_mean = means[source_idx]  # (B, D)
            mean_shift = key_mean - current_mean

            z_cf = z_batch + mean_shift
            x_cf, _ = analyzer.model(z_cf, rev=True)

            all_x_cf.append(x_cf.cpu())
            all_z_cf.append(z_cf.cpu())
            all_source_labels.extend(source_labels)

    all_x_cf = torch.cat(all_x_cf, dim=0)
    all_z_cf = torch.cat(all_z_cf, dim=0)

    return all_x_cf, all_z_cf, all_source_labels, key


def generate_counterfactuals_scgen(analyzer, key):
    """scGen counterpart to generate_counterfactuals: for every other condition
    ("source"), uses SCGEN.predict (official ctrl -> stim latent-arithmetic) to
    predict what each source cell would look like under `key`.

    analyzer.model must be a trained scgen.SCGEN (or SCGENMixturePriorWrapper around
    one) registered via `setup_anndata(adata, batch_key=<condition column>,
    labels_key=<cell type column>)`, since `.predict` reads the condition from the
    registered batch_key. analyzer.labels_key must match that batch_key.

    analyzer.adata holds the same raw counts scGen was trained/registered on, so
    `.predict`/`.get_latent_representation` are called unconverted. The returned x_cfs
    is log1p'd before returning (clamped to 0 first, since scGen's decoder has no
    positivity constraint and can predict negative values) so it's comparable to the
    log-space INN/scVI pipeline - see mmd.ctrl_mmd_dists/mmd_dists (log1p_transform=True)
    for how the "real" data it gets compared against is logged to match.
    """
    adata = analyzer.adata
    labels_key = analyzer.labels_key

    cats = adata.obs[f"{labels_key}"].astype("category").cat.categories.tolist()
    if key not in cats:
        raise ValueError(f"key '{key}' not found in adata.obs['{labels_key}']")

    all_x_cf = []
    all_z_cf = []
    all_source_labels = []

    for source in cats:
        if source == key:
            continue

        adata_source = adata[adata.obs[f"{labels_key}"] == source].copy()
        if adata_source.n_obs == 0:
            continue

        pred_adata, delta = analyzer.model.predict(
            ctrl_key=source,
            stim_key=key,
            adata_to_predict=adata_source,
        )

        z_source = analyzer.model.get_latent_representation(adata_source)
        x_cf_log = np.log1p(np.clip(pred_adata.X, a_min=0, a_max=None))

        all_x_cf.append(torch.tensor(x_cf_log, dtype=torch.float32))
        all_z_cf.append(torch.tensor(z_source + delta, dtype=torch.float32))
        all_source_labels.extend([source] * adata_source.n_obs)

    x_cfs = torch.cat(all_x_cf, dim=0)
    z_cfs = torch.cat(all_z_cf, dim=0)

    return x_cfs, z_cfs, all_source_labels, key


def init_classifier(analyzer, key, binary_key=None, latent=False, hidden_dim=None, labels_key=None):

    if labels_key is None:
        labels_key = analyzer.labels_key

    if latent:
        adata_train = create_latent_adata(
            analyzer.adata,
            analyzer.model,
            analyzer.device,
            analyzer.conditions,
            analyzer.condition_type,
        )
    else:
        adata_train = analyzer.adata.copy()

    if binary_key is not None:
        adata_train.obs[f"{labels_key}"] = adata_train.obs[f"{labels_key}"].apply(
            lambda x: f"{key}" if x == f"{key}" else "rest"
        )

    train_dataset, train_dataloader, test_dataset, test_dataloader = prepare_train_test_data(
        adata_train,
        batchsize=128,
        device=analyzer.device,
        dtype=analyzer.dtype,
        label_key=[f"{labels_key}"],
        test_size=0.2,
    )

    input_dim = adata_train.X.shape[1]
    num_classes = len(adata_train.obs[f"{labels_key}"].unique())

    classifier = LogisticRegressionClassifier(
        input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim
    ).to(analyzer.device)

    return classifier, train_dataloader, test_dataloader, train_dataset, adata_train, num_classes


def evaluate_training_error(analyzer, adata_train, labels_key, num_classes, class_accuracies):

    if labels_key is None:
        labels_key = analyzer.labels_key

    keys = adata_train.obs[f"{labels_key}"].value_counts().index.tolist()
    df = pd.DataFrame(
        {
            "Class": [f"{keys[i]}" for i in range(num_classes)],
            "Accuracy": class_accuracies,
        }
    )
    df.style.background_gradient(cmap="viridis", subset=["Accuracy"]).format({"Accuracy": "{:.2f}"})

    display(df)


def evaluate_cfs_classification(
    analyzer,
    classifier,
    key,
    train_dataset,
    latent=False,
    labels_key=None,
    cf_fn=generate_counterfactuals,
):

    if labels_key is None:
        labels_key = analyzer.labels_key

    x_cfs, z_cfs, all_source_labels, key_mean = cf_fn(analyzer, key=key)
    adata_cf = analyzer.adata[analyzer.adata.obs[analyzer.labels_key] != key].copy()

    if latent:
        adata_cf.X = z_cfs.numpy()
    else:
        adata_cf.X = x_cfs.numpy()

    cf_dataset = AdataDataset(
        adata_cf,
        label_key=None,
        device=analyzer.device,
        dtype=analyzer.dtype,
    )

    cf_loader = DataLoader(
        cf_dataset,
        batch_size=256,
        shuffle=False,
    )

    cats = train_dataset.cats[0].categories.tolist()
    target_idx = cats.index(key)

    classifier.eval()

    predictions = []
    probabilities = []

    with torch.no_grad():
        for x, _ in cf_loader:
            x = x.to(analyzer.device)

            logits = classifier(x)
            probs = torch.softmax(logits, dim=1)

            predictions.append(torch.argmax(probs, dim=1).cpu())
            probabilities.append(probs.cpu())

    predictions = torch.cat(predictions)
    probabilities = torch.cat(probabilities)

    results = []

    source_labels = np.asarray(all_source_labels)

    for source in np.unique(source_labels):
        mask = source_labels == source

        frac = (predictions[mask] == target_idx).float().mean().item()
        mean_prob = probabilities[mask, target_idx].mean().item()

        results.append(
            {
                "source": source,
                "fraction_predicted_target": frac,
                "mean_target_probability": mean_prob,
                "n_cells": int(mask.sum()),
            }
        )

    results = pd.DataFrame(results).sort_values("fraction_predicted_target", ascending=False)

    print(results)

    return x_cfs, z_cfs, all_source_labels, target_idx
