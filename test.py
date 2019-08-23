import colorama
import torch

from data import create_loader
from model import create_model
from test_arguments import Arguments
from utils import calculate_f1_score, multiclass_roc_auc_score

# setting up the colors:
reset = colorama.Style.RESET_ALL
blue = colorama.Fore.BLUE
red = colorama.Fore.RED
green = colorama.Fore.GREEN

args = Arguments().parse()

args.phase = 'test'


data_loader, weights = create_loader(args)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

nl = '\n'
print(f'{blue}There are a total number of {red}{dataset_size}{blue} frames in the positive test data set.{reset}{nl}')

model = create_model(args, weights, data_loader.dataset.classes)
model.set_up(args)

print(f'{nl}{red}Processing the positive test images has begun..{reset}{nl}')

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
# print(accuracy, dataset_size)
# print(f'gt.len: {len(gts)}')
# print(f'pred.len: {len(preds)}')
f1 = calculate_f1_score(gts, preds)
auc = multiclass_roc_auc_score(gts, preds)

print(f'Accuracy: {green}{accuracy:.4f}{reset} -- F1 Score: {green}{f1:.4f}{reset} -- AUC: {green}{auc:.4f}{reset}')
