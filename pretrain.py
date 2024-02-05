from LSTM import MyLSTM,Single_Variable_LSTM
import torch
import pandas as pd
import numpy as np


def train_and_save_model(train_y, prev_days_for_train, epochs, lrr, device):
    model = Single_Variable_LSTM(train_y, prev_days_for_train, device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.lstm.parameters(), lr=lrr)

    num_epochs = epochs

    for epoch in range(num_epochs):
        outputs = model.lstm(model.train_x_tensors)  # 前向传播
        optimizer.zero_grad()  # 梯度归零
        loss = criterion(outputs, model.train_y_tensors)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # 保存预训练权重
    torch.save(model.lstm.state_dict(), 'pretrained_lstm.pth')
    print("Model saved as pretrained_lstm.pth")


# train_and_save_model(train_y, prev_days_for_train, "cuda:1")


def main():
    prev_days_for_train = 50
    range_lines=(0,7283)

    data_all=pd.read_csv("data_w_mmtm.csv")
    data=data_all[["DATA_NUM","new_mmtm1","new_mmtm2","point_no"]]
    windows=5
    p_new_mmtm1=pd.DataFrame(data["new_mmtm1"][range_lines[0]:range_lines[1]])
    rp_new_mmtm1=p_new_mmtm1.rolling(window=windows).mean()
    r_new_mmtm1=rp_new_mmtm1.values.tolist()
    p_new_mmtm2=pd.DataFrame(data["new_mmtm2"][range_lines[0]:range_lines[1]])
    rp_new_mmtm2=p_new_mmtm2.rolling(window=windows).mean()
    r_new_mmtm2=rp_new_mmtm2.values.tolist()
    for i in range(0,windows-1):
        r_new_mmtm1[i][0]=0
        r_new_mmtm2[i][0]=0

    x = np.array(data["point_no"][range_lines[0]:range_lines[1]])
    r_delta=[(x[0]-y[0]) for x, y in list(zip(r_new_mmtm1,r_new_mmtm2))]

    y = np.array(r_delta)

    train_and_save_model(y, prev_days_for_train, epochs=2000, lrr=0.003, device="cuda:1")
    


if __name__ == '__main__':
    main()