import torch.nn as nn
import torch
from tqdm import tqdm


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
    device,
    dtype=torch.float32,
    num_epochs=20,
    learning_rate=1e-3,
):
    criterion = nn.CrossEntropyLoss()
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
