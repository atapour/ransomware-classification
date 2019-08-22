import time
import colorama
from train_arguments import Arguments
from data import create_loader
from model import create_model
from utils import Display

# setting up the colors:
reset = colorama.Style.RESET_ALL
green = colorama.Fore.GREEN
red = colorama.Fore.RED

args = Arguments().parse()
data_loader, weights = create_loader(args)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

nl = '\n'
print(f'{green}There are a total number of {red}{dataset_size}{green} images in the data set.{reset}{nl}')

model = create_model(args, weights, data_loader.dataset.classes)
model.set_up(args)
display = Display(args)
global_step = 0
total_steps = 0

while global_step < args.num_steps:
    data_time_start = time.time()

    for j, data in enumerate(data_loader):
        processing_time_start = time.time()

        if global_step % args.print_freq == 0:
            t_data = processing_time_start - data_time_start


        model.assign_inputs(data)
        model.optimize(args)

        if global_step % args.display_freq == 0:
            display.display_current_results(model.get_train_images())

        if global_step % args.print_freq == 0:
            loss = model.get_loss()
            t_proc = (time.time() - processing_time_start) / args.batch_size
            display.print_current_loss(global_step, loss, t_proc, t_data)
            if args.display_id > 0:
                display.plot_current_loss(global_step, loss)


        if total_steps % args.save_checkpoint_freq == 0:
            print('saving the latest model (total_steps %d)' % (total_steps))
            model.save_networks(total_steps)

        global_step += 1
        total_steps += args.batch_size
