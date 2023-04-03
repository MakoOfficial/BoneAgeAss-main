
import myKit
if __name__ == '__main__':
    data_dir = '../data/archive/'
    num_epochs, lr, wd = 100, 2e-4, 5e-4
    lr_period, lr_decay = 10, 0.5
    data = myKit.read_all_image(rootPath='../data/archive/two_label')
    Id, age, gender = myKit.read_labels(data_dir, 'two_label.csv')
    # data = data[:100, :]
    # age = age[:100, :]
    # gender = gender[:100, :]
    # weight = myKit.get_weight(age)
    # print(weight)
    # data_train, age_train, gender_train, data_valid, age_valid, gender_valid = myKit.get_k_fole_data(10, 0, data, age, gender)
    net = myKit.get_net(isEnsemble=False)
    myKit.two_label(net, data, age, gender, num_epochs, lr, 10, lr_period, lr_decay)