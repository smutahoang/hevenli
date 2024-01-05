import random
import pickle
from trainers import train_eval
from models import MonoLingualModel, MultiLingualModel, SiameseModel
import torch
import json
import copy
import sys


def measure_performance(model_class, data):
    random.seed(0)
    random.shuffle(data)

    fold_size = len(data) // 5
    performance = []
    for i in range(5):
        train_data = data[:i * fold_size] + data[(i + 1) * fold_size:]
        test_data = data[i * fold_size:(i + 1) * fold_size]
        print('experimenting with fold ', i)
        model = copy.deepcopy(model_class)
        trace_performance = train_eval(model, train_data, test_data, num_epochs=5, batch_size=16)
        performance.append(trace_performance)
    return performance


def examine(model, data):
    random.seed(0)
    random.shuffle(data)
    splitting_point = int(len(data) * 0.8)
    train_data = data[:splitting_point]
    test_data = data[splitting_point:]

    train_eval(model, train_data, test_data)


def examine_all_models():
    device = torch.device('cuda:0')
    #     file = open('larger_tokenized_health_en_vi.json', 'r')
    file = open('heavenli.json', 'r')
    data = [json.loads(line.strip()) for line in file]
    file.close()

    #     # monolingual-base
    #     print("monolingual-base")
    #     model = MonoLingualModel(en_pretrained_model='roberta-base',
    #                              vi_pretrained_model="vinai/phobert-base",
    #                              device=device)
    #     model = model.to(device)
    #     performance = measure_performance(model, data)
    #     pickle.dump(performance, open('monolingual_base.pkl', 'wb'))

    #     # monolingual-large
    #     print("monolingual-large")
    #     model = MonoLingualModel(en_pretrained_model='roberta-large',
    #                              vi_pretrained_model="vinai/phobert-large",
    #                              device=device)
    #     model = model.to(device)
    #     performance = measure_performance(model, data)
    #     pickle.dump(performance, open('monolingual_large.pkl', 'wb'))

    #     # multilingual-mbert-base
    #     print("multilingual-mbert-base")
    #     pretrained_model = 'bert-base-multilingual-cased'
    #     model = MultiLingualModel(pretrained_model=pretrained_model, device=device)
    #     model = model.to(device)
    #     performance = measure_performance(model, data)
    #     pickle.dump(performance, open('multilingual_mbert_base.pkl', 'wb'))

    #     # multilingual-mbert-large
    #     # pretrained_model = 'bert-large-multilingual-cased'
    #     # model = MultiLingualModel(pretrained_model=pretrained_model, device=device)
    #     # model = model.to(device)
    #     # performance = measure_performance(model, data)
    #     # pickle.dump(performance, open('multilingual_mbert_large.pkl', 'wb'))

    #     # multilingual-xlm-base
    #     print("multilingual-xlm-base")
    #     pretrained_model = 'xlm-roberta-base'
    #     model = MultiLingualModel(pretrained_model=pretrained_model, device=device)
    #     model = model.to(device)
    #     performance = measure_performance(model, data)
    #     pickle.dump(performance, open('multilingual_xlmr_base.pkl', 'wb'))

    #     # multilingual-xlm-large
    #     print("multilingual-xlm-large")
    #     pretrained_model = 'xlm-roberta-large'
    #     model = MultiLingualModel(pretrained_model=pretrained_model, device=device)
    #     model = model.to(device)
    #     performance = measure_performance(model, data)
    #     pickle.dump(performance, open('multilingual_xlmr_large.pkl', 'wb'))

    #     multilingual-mdeberta-base
    print("multilingual-mdeberta-base")
    pretrained_model = 'microsoft/deberta-v3-base'
    model = MultiLingualModel(pretrained_model=pretrained_model, device=device)
    model = model.to(device)
    performance = measure_performance(model, data)
    pickle.dump(performance, open('multilingual_deberta_base.pkl', 'wb'))

    # multilingual-mdeberta-large
    print("multilingual-mdeberta-large")
    pretrained_model = 'microsoft/deberta-v3-large'
    model = MultiLingualModel(pretrained_model=pretrained_model, device=device)
    model = model.to(device)
    performance = measure_performance(model, data)
    pickle.dump(performance, open('multilingual_deberta_large.pkl', 'wb'))

    # siamese-base
    print("siamese-base")
    model = SiameseModel(model='base', device=device)
    model = model.to(device)
    performance = measure_performance(model, data)
    pickle.dump(performance, open('siamese_base.pkl', 'wb'))

    # siamese-large
    print("siamese-large")
    model = SiameseModel(model='large', device=device)
    model = model.to(device)
    performance = measure_performance(model, data)
    pickle.dump(performance, open('siamese_large.pkl', 'wb'))


def create_model(utilization, size, device):
    # monolingual-base
    print(f'utilization = {utilization} size = {size}')

    if utilization == 'mono' and size == 'base':
        model = MonoLingualModel(en_pretrained_model='roberta-base',
                                 vi_pretrained_model="vinai/phobert-base", device=device)

    # monolingual-large
    if utilization == 'mono' and size == 'large':
        model = MonoLingualModel(en_pretrained_model='roberta-large',
                                 vi_pretrained_model="vinai/phobert-large", device=device)

    # multilingual-mbert-base
    if utilization == 'multi_mbert' and size == 'base':
        pretrained_model = 'bert-base-multilingual-cased'
        model = MultiLingualModel(pretrained_model=pretrained_model, device=device)

    # multilingual-xlm-base
    if utilization == 'multi_xlmr' and size == 'base':
        pretrained_model = 'xlm-roberta-base'
        model = MultiLingualModel(pretrained_model=pretrained_model, device=device)

    # multilingual-xlm-large
    if utilization == 'multi_xlmr' and size == 'large':
        pretrained_model = 'xlm-roberta-large'
        model = MultiLingualModel(pretrained_model=pretrained_model, device=device)

    #     multilingual-mdeberta-base
    if utilization == 'multi_deberta' and size == 'base':
        pretrained_model = 'microsoft/deberta-v3-base'
        model = MultiLingualModel(pretrained_model=pretrained_model, device=device)

    # multilingual-mdeberta-large
    if utilization == 'multi_deberta' and size == 'large':
        pretrained_model = 'microsoft/deberta-v3-large'
        model = MultiLingualModel(pretrained_model=pretrained_model, device=device)

    # siamese-base
    if utilization == 'siamese' and size == 'base':
        model = SiameseModel(model='base', device=device)

    # siamese-large
    if utilization == 'siamese' and size == 'large':
        model = SiameseModel(model='large', device=device)

    return model


def examine_a_model(dataset, utilization, size, batch_size, gpu_index):
    print("**************************************************************************************")
    print('utilization = ', utilization)
    print('size = ', size)
    print('batch_size = ', batch_size)

    file = open(f'{dataset}.json', 'r')
    data = [json.loads(line.strip()) for line in file]
    file.close()

#    data = data[:100]

    random.seed(0)
    random.shuffle(data)

    fold_size = len(data) // 5
    performance = []
    for i in range(5):
        train_data = data[:i * fold_size] + data[(i + 1) * fold_size:]
        test_data = data[i * fold_size:(i + 1) * fold_size]
        print('experimenting with fold ', i)
        device = torch.device(f'cuda:{gpu_index}')
        model = create_model(utilization=utilization, size=size, device=device)
        model = model.to(device)
        trace_performance = train_eval(model, train_data, test_data, num_epochs=5, batch_size=batch_size)
        performance.append(trace_performance)
        pickle.dump(performance, open(f'{dataset}_{utilization}_{size}.pkl', 'wb'))



if __name__ == "__main__":
    # examine_all_models()
    dataset = sys.argv[1]
    utilization = sys.argv[2]
    size = sys.argv[3]
    batch_size = int(sys.argv[4])
    gpu_index = int(sys.argv[5])
    examine_a_model(dataset=dataset, utilization=utilization, size=size, batch_size=batch_size, gpu_index=gpu_index)
