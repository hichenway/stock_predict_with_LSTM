import numpy as np

# pytorch 模型
import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset

class Net(Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                         num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
        self.linear = Linear(in_features=config.hidden_size, out_features=config.output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)     # LSTM 的返回很多
        linear_out = self.linear(lstm_out)
        return linear_out

def train(config, train_X, train_Y, valid_X, valid_Y):
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    model = Net(config)
    if config.add_train:
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()

    valid_loss_min = float("inf")
    bad_epoch = 0
    for epoch in range(config.epoch):
        print("Current epoch: ", epoch)
        model.train()
        train_loss_array = []
        for i, _data in enumerate(train_loader):
            data_X, data_Y = _data
            optimizer.zero_grad()
            pred_X = model(data_X)
            loss = criterion(pred_X, data_Y)
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.item())

        # 以下为早停机制
        model.eval()
        valid_loss_cur = np.mean([criterion(model(_X), _Y).item() for _X, _Y in valid_loader])
        print("The train loss is {:.4f}. ".format(np.mean(train_loss_array)),
              "The valid loss is {:.4f}.".format(valid_loss_cur))

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.model_name)
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:
                print(" The training stops early in epoch".format(epoch))
                break



def predict(config, test_X):
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    model = Net(config)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))

    result = torch.Tensor()

    model.eval()
    for _data in test_loader:
        data_X = _data[0]
        pred_X = model(data_X)
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().numpy()