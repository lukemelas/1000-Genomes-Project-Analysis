import torch
from torch.autograd import Variable as V
use_gpu = torch.cuda.is_available()

import pdb

def validate(model, val_loader, criterion, visualize=False):
    model.eval()

    total = 0
    total_correct = 0
    total_loss = 0

    if visualize: # store results for confusion matrix
        results = []
        results.append([])
        results.append([])
    else:
        results = None

    for input, label in val_loader:
        input = V(input, volatile=True).cuda() if use_gpu else V(input, volatile=True)
        label = V(label, volatile=True).cuda() if use_gpu else V(label, volatile=True)
        output = model(input)
        _, pred = torch.max(output, 1)
        total_correct += (pred == label).long().data.sum()
        loss = criterion(output, label)
        total_loss += loss.data[0]
        total += label.numel()
    
        if visualize:
            results[0].append(pred)
            results[1].append(label)

    return total_correct, total, results

def train(model, train_loader, val_loader, optimizer, criterion, logger, num_epochs=30, print_freq=50, model_id=1):
    best_acc = 0
    vis_accs = []
    for epoch in range(num_epochs):
        model.train()

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

            logger.log('Epoch[{}/{}] \t Iter [{:>3}/{:>3}] \t Loss: {:.3f}'.format(
                epoch + 1, num_epochs, (i+1), len(train_loader), total_loss/total), stdout=False)

        # One epoch of validation
        total_correct, total, _ = validate(model, val_loader, criterion)
        acc = total_correct / total

        # Save best model
        if acc > best_acc:
            best_acc = acc
            logger.save_model(model.state_dict(), 'model_{}.pth'.format(model_id))
            logger.log('Best accuracy: {:.3f}'.format(best_acc), stdout=False)

        # Log
        logger.log('Epoch[{}/{}] \t Validation Accuracy {}/{} = {:.3f}% \t Best Accuracy: {:.3f}'.format(
            epoch + 1, num_epochs, total_correct, total, 100 * acc, best_acc), stdout=(epoch % print_freq == 0))

        # For visualization
        vis_accs.append((total_loss/total, acc))

    # Save for visualization
    logger.save_model(vis_accs, 'visualization_accuracies.pt')

    return best_acc


