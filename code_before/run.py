
import myKit
import ToolKit
import torch
if __name__ == '__main__':
    data_dir = '../data/archive/'
    num_epochs, lr, wd = 100, 1e-4, 5e-4
    lr_period, lr_decay = 10, 0.5
    data = myKit.read_all_image(rootPath='../data/archive/small-dataset')
    # Id, age, gender = myKit.read_labels(data_dir, 'small-dataset.csv')
    age, gender = ToolKit.read_label(filePath='../data/archive/small-dataset.csv', record_num=len(data))
    data = data.type(torch.FloatTensor)
    gender = gender.type(torch.FloatTensor)
    age = age.type(torch.FloatTensor)
    batch_size = 10
    # data = data[:100, :]
    # age = age[:100, :]
    # gender = gender[:100, :]
    # weight = myKit.get_weight(age)
    # print(weight)
    # data_train, age_train, gender_train, data_valid, age_valid, gender_valid = myKit.get_k_fole_data(10, 0, data, age, gender)
    # train_l, valid_l = myKit.k_fold(10, data, age, gender, num_epochs, lr, wd, 10, lr_period, lr_decay)
    # print(train_l, valid_l)
    # myKit.k_fold(k=10, train_data=data, train_age=age, train_gender=gender, num_epochs=num_epochs, learning_rate=lr, weight_decay=wd, batch_size=batch_size, lr_period=lr_period, lr_decay=lr_decay)
    net = myKit.get_net(isEnsemble=False)
    # net.double() 只能用float，不然显存就爆了
    net = net.to(device=myKit.try_gpu())
    net.fine_tune()
    myKit.two_label(net=net, train_data=data, train_age=age, train_gender=gender, num_epochs=num_epochs, learning_rate=lr, batch_size=batch_size, lr_period=lr_period, lr_decay=lr_decay)
    # module = ToolKit.RunModuleTrainBatch(module=net, train_data=data, train_sex=gender, real_label=age, batch_size=batch_size, lr=lr, EPOCH=num_epochs)