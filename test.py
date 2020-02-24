import torch

from data import create_loader
from model import create_model
from test_arguments import Arguments
from utils import calculate_f1_score, multiclass_roc_auc_score

args = Arguments().parse()

args.phase = 'test'

data_loader, weights = create_loader(args)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

nl = '\n'
print(f'There are a total number of {dataset_size} frames in the positive test data set.{nl}')

model = create_model(args, weights, data_loader.dataset.classes)
model.set_up(args)

print(f'{nl}Processing the positive test images has begun..{nl}')

gts = []
preds = []
accuracy = 0.0

with torch.no_grad():
        for j, data in enumerate(data_loader):

                model.assign_inputs(data)
                model.test()
                output = model.get_test_outputs()

                gt = output['gt']
                out = output['out']

                gts.append(gt.item())
                preds.append(out.item())

                accuracy += out.eq(gt).float().mean()

accuracy /= dataset_size
f1 = calculate_f1_score(gts, preds)
auc = multiclass_roc_auc_score(gts, preds)

print(f'Accuracy: {accuracy:.4f} -- F1 Score: {f1:.4f} -- AUC: {auc:.4f}')
