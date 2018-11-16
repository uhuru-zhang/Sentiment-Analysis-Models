from collections import Counter

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D

from models.data_set.apsect_level_chinese_restaurant_reviews import TextDataSet
from models.models.attention_based_lstm.model import Model

from tensorboardX import SummaryWriter

class_num = 4

if __name__ == '__main__':
    writer = SummaryWriter()

    train_loader = D.DataLoader(TextDataSet(train=True, columns=["content_indexes", "dish_taste"]),
                                batch_size=256,
                                shuffle=True, num_workers=32)

    test_loader = D.DataLoader(TextDataSet(train=False, columns=["content_indexes", "dish_taste"]),
                               batch_size=256,
                               shuffle=True, num_workers=32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(vocabulary_size=340166, device=device).to(device)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    optimizer = optim.Adadelta(params=model.parameters())

    globals_index = 0
    for epoch in range(1, 100):
        train_loss, train_accuracy = 0, 0
        model.train()
        for batch_index, line in enumerate(train_loader):
            contents, aspect_level = line[0], line[1]

            b_content_indexes = [[int(index_str) for index_str in content.split(" ")] for content in contents]

            index_lengths = [(i, len(content_indexes)) for i, content_indexes in enumerate(b_content_indexes)]

            lengths = [length for _, length in sorted(index_lengths, key=lambda a: a[1], reverse=True)]
            indexes = [index for index, _ in sorted(index_lengths, key=lambda a: a[1], reverse=True)]

            content_indexes_padding = [content_indexes + (lengths[0] - len(content_indexes)) * [340165] for
                                       content_indexes in
                                       b_content_indexes]

            data = torch.tensor(content_indexes_padding, dtype=torch.long)
            data = torch.index_select(data, dim=0, index=torch.Tensor(indexes).long()).to(device)

            target = torch.tensor([environment_cleaness + 2 for environment_cleaness in aspect_level])
            target = torch.index_select(target, dim=0, index=torch.Tensor(indexes).long()).to(device)

            aspect = torch.Tensor([170]).long().to(device)
            optimizer.zero_grad()

            output = model(data, lengths, aspect)

            loss = F.cross_entropy(input=output, target=target)
            loss.backward()
            optimizer.step()

            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.view_as(pred)).sum().item()

            train_loss, train_accuracy = loss.item(), 100. * correct / len(contents)
            if batch_index % 100 == 0:
                print("Train Epoch: {} [{}/{} ({:.0f})]\tLoss: {}\t, Accuracy: ({:.2f}%)".format(
                    epoch, batch_index * len(data), len(train_loader.dataset),
                           100. * batch_index / len(train_loader), loss.item(),
                           100. * correct / len(contents)
                ))
            print("train epoch: {}, batch_index: {}, loss: {}, Accuracy: ({:.2f}%)".format(epoch, batch_index, train_loss,
                                                                                           train_accuracy))
            writer.add_scalar("train/accuracy", train_accuracy, global_step=globals_index)
            writer.add_scalar("train/loss", train_loss, global_step=globals_index)
            globals_index += 1

        print("epoch {} done!".format(epoch))

        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0
            result = Counter()
            types = Counter()
            pred_types = Counter()

            for batch_index, line in enumerate(test_loader):
                print("test epoch, batch_index:", epoch, batch_index)
                contents, aspect_level = line[0], line[1]

                b_content_indexes = [[int(index_str) for index_str in content.split(" ")] for content in contents]

                index_lengths = [(i, len(content_indexes)) for i, content_indexes in enumerate(b_content_indexes)]

                lengths = [length for _, length in sorted(index_lengths, key=lambda a: a[1], reverse=True)]
                indexes = [index for index, _ in sorted(index_lengths, key=lambda a: a[1], reverse=True)]

                content_indexes_padding = [content_indexes + (lengths[0] - len(content_indexes)) * [340165] for
                                           content_indexes in
                                           b_content_indexes]

                data = torch.tensor(content_indexes_padding, dtype=torch.long)
                data = torch.index_select(data, dim=0, index=torch.Tensor(indexes).long()).to(device)

                target = torch.tensor([environment_cleaness + 2 for environment_cleaness in aspect_level])
                target = torch.index_select(target, dim=0, index=torch.Tensor(indexes).long()).to(device)
                aspect = torch.Tensor([170]).long().to(device)

                output = model(data, lengths, aspect)

                test_loss += F.cross_entropy(input=output, target=target)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

                pred_target_tuple = [(p.item(), t.item()) for p, t in zip(pred, target)]

                for i in range(class_num):
                    result["TP_{}".format(i)] += sum(
                        [1 for _ in filter(lambda items: items[0] == i and items[1] == i, pred_target_tuple)])
                    result["FP_{}".format(i)] += sum(
                        [1 for _ in filter(lambda items: items[0] == i and items[1] != i, pred_target_tuple)])
                    result["FN_{}".format(i)] += sum(
                        [1 for _ in filter(lambda items: items[0] != i and items[1] == i, pred_target_tuple)])
                    result["TF_{}".format(i)] += sum(
                        [1 for _ in filter(lambda items: items[0] != i and items[1] != i, pred_target_tuple)])

                    types[i] += sum([1 for _ in filter(lambda items: items[1] == i, pred_target_tuple)])
                    pred_types[i] += sum(
                        [1 for _ in filter(lambda items: items[0] == items[1] == i, pred_target_tuple)])

            F1 = []
            for i in range(class_num):
                try:
                    precision = result["TP_{}".format(i)] / (result["TP_{}".format(i)] + result["FP_{}".format(i)])
                    recall = result["TP_{}".format(i)] / (result["TP_{}".format(i)] + result["FN_{}".format(i)])
                    F1.append((i, 2 * (precision * recall) / (precision + recall)))
                except:
                    F1.append((i, 0))
                    print(pred_types)
                    print(types)
                    print(result)
            F1_ave = sum(map(lambda items: items[1], F1)) / len(F1)

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), F1: {}, F1_AVE: {} \n'
                  .format(test_loss, correct, len(test_loader.dataset),
                          100. * correct / len(test_loader.dataset), F1, F1_ave))

            f_scalars = {str(i - 2): f1 for i, f1 in F1}
            f_scalars["avg"] = F1_ave

            writer.add_scalars("F1", f_scalars, epoch)
            writer.add_scalars("accuracy", {"train": train_accuracy, "test": correct / len(test_loader.dataset)}, epoch)
            writer.add_scalars("lost", {"train": train_loss, "test": test_loss}, epoch)
