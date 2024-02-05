from LSTM import MyLSTM,Single_Variable_LSTM
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def main():
    prev_days_for_train = 50
    range_lines=(6950,7283)

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

    y = np.array(r_delta[0:250])
    
    model = Single_Variable_LSTM(y, prev_days_for_train, epochs=3000, lr=0.001, device="cuda:1")
    pretrained_weights = torch.load('pretrained_lstm.pth', map_location=model.device)
    model.lstm.load_state_dict(pretrained_weights)
    y = y[prev_days_for_train:]

    a = model.train()
    b = model.predict(75)[2]
    result = np.concatenate((a, b), axis=0)
    result = result.reshape(1, -1)                                          
    result = np.squeeze(result)


    real_series = pd.Series(np.array(r_delta), name="Real")
    predict_index = np.arange(prev_days_for_train, prev_days_for_train + len(result))
    predict_series = pd.Series(result, index=predict_index, name="Predict")
    combined_df = pd.concat([real_series, predict_series], axis=1, join='outer')
    combined_df.to_csv("real_vs_predict.csv", index_label="Index")

    plt.plot(np.array(r_delta), label='real')
    plt.plot(np.arange(prev_days_for_train, prev_days_for_train+len(result)), result, label='predict')
    plt.axvline(x=len(y)+prev_days_for_train, color='r', linestyle='--')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()