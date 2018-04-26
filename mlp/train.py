import torch
from torch.autograd import Variable as V
use_gpu = torch.cuda.is_available()

def validate(model, val_loader, criterion):
    total = 0
    total_correct = 0
    total_loss = 0
    for input, label in val_loader:
        input = V(input, volatile=True).cuda() if use_gpu else V(input, volatile=True)
        label = V(label, volatile=True).cuda() if use_gpu else V(label, volatile=True)
        output = model(input)
        _, pred = torch.max(output, 1)
        total_correct += (pred == label).long().data.sum()
        loss = criterion(output, label)
        total_loss += loss.data[0]
        total += label.numel()
    return total_correct, total

def train(model, train_loader, val_loader, optimizer, criterion, logger, num_epochs=30, print_freq=50):
    accs = []
    best_acc = 0
    for epoch in range(num_epochs):

        # One epoch of training
        total = 0
        total_loss = 0
        for i, (input, label) in enumerate(train_loader):
            input = V(input).cuda() if use_gpu else V(input)
            label = V(label).cuda() if use_gpu else V(label)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.data[0]
            total += label.numel()

            if i % print_freq == 0:
                logger.log('Epoch[{}/{}] \t Iter [{:>3}/{:>3}] \t Loss: {:.3f}'.format(
                    epoch + 1, num_epochs, (i+1), len(train_loader), total_loss/total))

        # One epoch of validation
        total_correct, total = validate(model, val_loader, criterion)
        acc = total_correct / total
        accs.append(acc)
        logger.log('Epoch[{}/{}] \t Validation Accuracy {}/{} = {:.3f}% \n'.format(
            epoch + 1, num_epochs, total_correct, total, 100 * acc))

        # Save best model
        if epoch % 20 == 1 and acc > best_acc:
            best_acc = acc
            logger.save_model(model.state_dict())

    logger.log('Best accuracy: {:.3f}'.format(best_acc))
    return accs


