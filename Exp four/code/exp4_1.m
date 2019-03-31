train_data = load('training_data.txt');
test_data = load('test_data.txt');
[label_pre,sum,sucess_rate] = LogMLE(test_data,train_data);
