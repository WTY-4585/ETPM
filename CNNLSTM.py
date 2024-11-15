import torch
import torch.nn as nn
import numpy as np
from ultis.readdatalabel import *


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.6):
        super(LSTMClassifier, self).__init__()
        self.cnn = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dtype = torch.float32

    def forward(self, x):
        x = x.to(dtype=self.dtype)
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.squeeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out


def feature_explore(x):
    dx = np.gradient(x, axis=2)
    ix = np.cumsum(x, axis=2)
    x_3 = np.concatenate((x, dx, ix), axis=1)
    return torch.tensor(x_3)


datapercent = 0.25
input_size = 500
hidden_size = 32
num_layers = 2
num_classes = 30

device = torch.device('cuda')
model = LSTMClassifier(int(input_size*datapercent), hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

train_dataloader, test_dataloader = create_dataloaders(data_dir='./data/30class/', batch_size=256)

break_step = 100
train_losses = []
test_losses = []

print("Running training...")
for epoch in range(100):
    train_loss = 0
    train_samples = 0
    model.train()

    for step, (data, label) in enumerate(train_dataloader):
        data = feature_explore(data)
        data = data[:, :, :int(datapercent * data.size(2))]
        data = data.to(device)
        label = label.to(device).long()

        output = model(data)
        loss = criterion(output, label)
        train_loss += loss.item() * data.size(0)
        train_samples += data.size(0)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step > break_step:
            break

    avg_train_loss = train_loss / train_samples
    train_losses.append(avg_train_loss)

    model.eval()
    test_loss = 0
    test_samples = 0

    with torch.no_grad():
        for step, (data, label) in enumerate(test_dataloader):
            data = feature_explore(data)
            data = data[:, :, :int(datapercent * data.size(2))]
            data = data.to(device)
            label = label.to(device).long()

            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item() * data.size(0)
            test_samples += data.size(0)

    avg_test_loss = test_loss / test_samples
    test_losses.append(avg_test_loss)

    print("Epoch {} train loss: {:.4f} test loss: {:.4f}".format(epoch, avg_train_loss, avg_test_loss))

model.eval()
test_loss = 0
correct_total = 0
total_samples = 0
class_correct = [0] * num_classes
class_total = [0] * num_classes

true_labels = []
predicted_labels_list = []

with torch.no_grad():
    for data, label in test_dataloader:
        data = feature_explore(data)
        data = data[:, :, :int(datapercent * data.size(2))]
        data = data.to(device)
        label = label.to(device).long()

        output = model(data)
        test_loss += criterion(output, label).item()

        true_labels.extend(label.cpu().numpy())
        predicted_labels_list.extend(output.argmax(dim=1).cpu().numpy())

        _, predicted = torch.max(output, 1)
        correct_total += (predicted == label).sum().item()
        total_samples += label.size(0)

        for i in range(num_classes):
            class_mask = (label == i)
            class_correct[i] += (predicted[class_mask] == label[class_mask]).sum().item()
            class_total[i] += class_mask.sum().item()

avg_test_loss = test_loss / len(test_dataloader)
overall_accuracy = correct_total / total_samples

print("Overall Test Loss: {:.4f}".format(avg_test_loss))
print("Overall Accuracy: {:.4f}".format(overall_accuracy))

for i in range(num_classes):
    class_accuracy = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print("Class {} Accuracy: {:.4f}".format(i, class_accuracy))





