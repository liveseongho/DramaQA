import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def accuracy_diffN(output, target, difficulties, N):
    _, output_idx = output.max(dim=1)

    assert output.shape[0] == len(target)
    assert len(difficulties) == len(target)

    results = output_idx == target
    correct = 0
    n = 0

    for i, diff in enumerate(difficulties):
        if diff == N:
            if results[i]:
                correct += 1
            n += 1
    if n == 0:
        r = 0
    else:
        r = correct / n
    return r, n


def accuracy_diff1(output, target, difficulties):
    return accuracy_diffN(output, target, difficulties, 1)


def accuracy_diff2(output, target, difficulties):
    return accuracy_diffN(output, target, difficulties, 2)


def accuracy_diff3(output, target, difficulties):
    return accuracy_diffN(output, target, difficulties, 3)


def accuracy_diff4(output, target, difficulties):
    return accuracy_diffN(output, target, difficulties, 4)