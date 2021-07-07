from util.TwoDPCA import *
from util.TTwoPCA import *
from util.data_process import *
from util.model import *
from util.waivelet import *
from util.SAE import *
from train import *
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def one_model_evaluate(x,y,z,data,model_func,weights,title,ylabel):
    X_train, y_train, xVal, yVal, sc, X_test, y_test = data
    y_true = sc.inverse_transform(y_test.reshape(-1, 1))
    model = model_func(X_train, y_train, xVal, yVal)
    model.load_weights(weights)
    XX = model.predict(X_test)
    y_pred = sc.inverse_transform(XX)
    MSE = mean_squared_error(y_true, y_pred)
    plt.subplot(x,y,z)
    plt.title("Model {}: MSE {}".format(title,MSE),fontsize=5)
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.ylabel(ylabel)
    return y_test,XX

def one_metric(RNN,GRU,LSTM,metrics_kind,func):
    print(metrics_kind,end = "\n\n")
    print("{}\t{}\t{}\t{}\t{}\t{}".format(" ", "Base_line", "PCA", "2DPCA", "2D2DPCA", "SAE"))
    print("RNN", end="\t")
    for data in RNN:
        y_test, XX = data
        print(round(func(y_test, XX),4), end="\t")
    print("\n")
    print("GRU",end = "\t")
    for data in GRU:
        y_test, XX = data
        print(round(func(y_test, XX),4), end="\t")
    print("\n")
    print("LSTM", end="\t")
    for data in LSTM:
        y_test, XX = data
        print(round(func(y_test, XX),4), end="\t")
    print("\n")
    print("\n")

def print_metrics(RNN,GRU,LSTM):
    one_metric(RNN, GRU, LSTM, "MSE", mean_squared_error)
    one_metric(RNN, GRU, LSTM, "MAE", mean_absolute_error)
    one_metric(RNN, GRU, LSTM, "R_Square", r2_score)






if __name__ == "__main__":
    df = pd.read_csv("./data/addedfeatures.csv")
    new_date = pd.to_datetime(df["date"], format='%Y%m%d', errors='ignore')
    df["date"] = new_date
    df.index = df.date
    price = df["PRC"]
    fig = plt.figure()
    #RNN
    RNN = []
    plt.figure(figsize=(28, 12))
    fig.subplots_adjust(hspace=.4)
    RNN.append(one_model_evaluate(5, 3, 1,get_baseline_data(df),get_RNN_model, "./model_weights/BASE_RNN.h5","RNN","Base"))
    RNN.append(one_model_evaluate(5, 3, 4, get_wt_pca_data(df), get_RNN_model, "./model_weights/PCA_RNN.h5","RNN","PCA"))
    RNN.append(one_model_evaluate(5, 3, 7, get_wt_2dpca_data(df), get_RNN_model, "./model_weights/2dPCA_RNN.h5","RNN","2DPCA"))
    RNN.append(one_model_evaluate(5, 3, 10, get_wt_2d2dpca_data(df), get_RNN_model, "./model_weights/2d2dPCA_RNN.h5","RNN","2D2DPCA"))
    RNN.append(one_model_evaluate(5, 3,13, load_SAE_data(df), get_RNN_model, "./model_weights/AE_RNN.h5","RNN","SAE"))

    #GRU
    GRU = []
    GRU.append(one_model_evaluate(5, 3, 2, get_baseline_data(df), get_GRU_model, "./model_weights/BASE_GRU.h5", "GRU", "Base"))
    GRU.append(one_model_evaluate(5, 3, 5, get_wt_pca_data(df), get_GRU_model, "./model_weights/PCA_GRU.h5", "GRU", "PCA"))
    GRU.append(one_model_evaluate(5, 3, 8, get_wt_2dpca_data(df), get_GRU_model, "./model_weights/2dPCA_GRU.h5", "GRU", "2DPCA"))
    GRU.append(one_model_evaluate(5, 3, 11, get_wt_2d2dpca_data(df), get_GRU_model, "./model_weights/2d2dPCA_GRU.h5", "GRU","2D2DPCA"))
    GRU.append(one_model_evaluate(5, 3, 14, load_SAE_data(df), get_GRU_model, "./model_weights/AE_GRU.h5", "GRU", "SAE"))

    #LSTM
    LSTM = []
    LSTM.append(one_model_evaluate(5, 3, 3, get_baseline_data(df), get_LSTM_model, "./model_weights/BASE_LSTM.h5", "LSTM", "Base"))
    LSTM.append(one_model_evaluate(5, 3, 6, get_wt_pca_data(df), get_LSTM_model, "./model_weights/PCA_LSTM.h5", "LSTM", "PCA"))
    LSTM.append(one_model_evaluate(5, 3, 9, get_wt_2dpca_data(df), get_LSTM_model, "./model_weights/2dPCA_LSTM.h5", "LSTM", "2DPCA"))
    LSTM.append(one_model_evaluate(5, 3, 12, get_wt_2d2dpca_data(df), get_LSTM_model, "./model_weights/2d2dPCA_LSTM.h5", "LSTM","2D2DPCA"))
    LSTM.append(one_model_evaluate(5, 3, 15, load_SAE_data(df), get_LSTM_model, "./model_weights/AE_LSTM.h5", "LSTM", "SAE"))
    plt.savefig("./figures/result.png")

    print_metrics(RNN,GRU,LSTM)


    # compared best models with random forest
    from sklearn.ensemble import RandomForestRegressor

    data, price = data_imputation_without_feature_selection(df)
    data, price, sc = normalize(data, price)
    data, price = data_transform(data, price, window_size=60)
    data = data.reshape(data.shape[0], -1)
    train, test, l_train, l_test = split_data(data, price)
    rf = RandomForestRegressor()
    rf.fit(train, l_train)
    p_test = rf.predict(test)
    y_pred = p_test
    y_true = l_test
    print("\n\nCompared best models with RandomForst\n\n")
    print("{}\t{}\t{}\t{}".format("","MSE","MAE","R_square"))
    print("PCA-GRU",end = "\t")
    y1,y2 = GRU[1]
    for func in [mean_squared_error,mean_absolute_error,r2_score]:
        print(round(func(y1,y2),4),end = "\t")
    print("\n")
    print("2D2DPCA-GRU", end="\t")
    y1, y2 = GRU[3]
    for func in [mean_squared_error, mean_absolute_error, r2_score]:
        print(round(func(y1, y2), 4), end="\t")
    print("\n")
    print("2D2DPCA-LSTM", end="\t")
    y1, y2 = LSTM[3]
    for func in [mean_squared_error, mean_absolute_error, r2_score]:
        print(round(func(y1, y2), 4), end="\t")
    print("\n")
    print("RandomForest",end = "\t")
    for func in [mean_squared_error, mean_absolute_error, r2_score]:
        print(round(func(y_true, y_pred), 4), end="\t")
    print("\n")


