import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from utils import loss_d, loss_g, save_generated_images
import random
import matplotlib.pyplot as plt
from dataset import DatasetImages
from models import Discriminator, Generator

def _init_():
    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists('output/'+args.exp_name):
        os.makedirs('output/'+args.exp_name)
    if not os.path.exists('output/'+args.exp_name+'/'+'models'):
        os.makedirs('output/'+args.exp_name+'/'+'models')


def main(args):
	path_to_images = args.dataset_path 
	path_to_dir = args.output_path

	coeff_train = 1.0 
	batch_size = args.batch_size 
	embedding_dim = args.embedding_dim 
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use gpu/cpu.
	print(f"Device: {device}")

	Path(f"{path_to_dir}/weights_dcgan").mkdir(exist_ok=True) #
	Path(f"{path_to_dir}/generated_images").mkdir(exist_ok=True) 

	lr_g = args.lr_g # 
	lr_d = args.lr_d #
	 
	epochs = args.epochs # 

	transform = transforms.Compose([
	    transforms.Grayscale(), # 
	    transforms.Resize((64, 64)), # 
	    transforms.ToTensor(), # 
	    transforms.Normalize((0.5), (0.5)) # 
	])

	paths_images = list(Path(path_to_images).iterdir()) # 
	random.shuffle(paths_images) # 
	paths_train = paths_images[:int(len(paths_images) * coeff_train)] # 
	names_train = [x.name for x in paths_train]

	data_train = DatasetImages(paths_train, transform=transform)
	train_loader = DataLoader(data_train, shuffle=True, batch_size=batch_size)

	generator = Generator(embedding_dim)
	discriminator = Discriminator()

	generator = generator.to(device)
	discriminator = discriminator.to(device)

	optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
	optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))


	for epoch in range(epochs):

	    total = 0

	    total_loss_g = 0
	    total_loss_d = 0

	    with tqdm(train_loader, desc=f"Epoch {epoch}", leave=False) as pbar:

	        for batch in pbar:

	            real_images = batch.to(device)
	            batch_size = len(real_images)

	            random_latent_vectors = torch.randn(batch_size, embedding_dim).to(device)

	            for _ in range(5):
	                optimizer_d.zero_grad()

	                with torch.no_grad():
	                    generated_images = generator(random_latent_vectors).to(device)

	                real_preds = discriminator(real_images)
	                fake_preds = discriminator(generated_images)

	                loss_discriminator = loss_d(device, real_preds, fake_preds)
	                loss_discriminator.backward()
	                optimizer_d.step()

	            optimizer_g.zero_grad()
	            random_latent_vectors = torch.randn(batch_size, embedding_dim).to(device)
	            generated_images = generator(random_latent_vectors)
	            
	            fake_preds = discriminator(generated_images)

	            loss_generator = loss_g(device, fake_preds)

	            loss_generator.backward()
	            optimizer_g.step()

	            total_loss_g += loss_generator.item() * batch_size
	            total_loss_d += loss_discriminator.item() * batch_size

	            total += batch_size

	            pbar.set_postfix({
	                "Loss d" : f"{loss_discriminator.item():.4f}",
	                "Loss g" : f"{loss_generator.item():.4f}",
	            })

	    loss_generator_result = total_loss_g / total
	    loss_discriminator_result = total_loss_d / total
	    
	    # проверка качества генератора
	    save_generated_images(epoch, generator, embedding_dim, examples=10, device=device)
	    
	    tqdm.write(f"Epoch: {epoch}, loss_g: {loss_generator_result}, loss_d: {loss_discriminator_result}")

	            
	torch.save(generator.state_dict(), path_to_dir + '/weights_dcgan/generator.pt')
	torch.save(discriminator.state_dict(), path_to_dir + '/weights_dcgan/discriminator.pt')


def compare_images(img1, img2):
    return torch.mean(torch.abs(img1 - img2)).item()

def test(args):
	print("Evaluating...")
	model = Generator(args.embedding_dim)

	model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))  

	model.eval() 

	path_to_images = args.dataset_path 
	path_to_dir = args.output_path

	batch_size = args.batch_size 
	embedding_dim = args.embedding_dim 
	coeff_train = 1.0 

	transform = transforms.Compose([
	    transforms.Grayscale(), # 
	    transforms.Resize((64, 64)), # 
	    transforms.ToTensor(), # 
	    transforms.Normalize((0.5), (0.5)) # 
	])

	paths_images = list(Path(path_to_images).iterdir()) # 
	random.shuffle(paths_images) # 
	paths_train = paths_images[:int(len(paths_images) * coeff_train)] # 
	names_train = [x.name for x in paths_train]

	data_train = DatasetImages(paths_train, transform=transform)

	data = DataLoader(data_train, shuffle=False, num_workers=8)

	r, c = 3, 5
	noise = torch.randn(r * c, embedding_dim, device='cpu')
	with torch.no_grad():
	    gen_imgs = model(noise).cpu()

	fig, axs = plt.subplots(r, c, figsize=(10, 6))
	fig.suptitle("Сгенерированные изображения")
	cnt = 0
	for i in range(r):
	    for j in range(c):
	        axs[i, j].imshow(gen_imgs[cnt, 0], cmap="gray")
	        axs[i, j].axis("off")
	        cnt += 1
	plt.show()

	# Поиск ближайших изображений из обучающего набора к сгенерированным.
	fig, axs = plt.subplots(r, c, figsize=(10, 6))
	fig.suptitle("Ближайшие изображения из обучающего набора")

	list_diffs = [0] * len(data)
	for ind, k in tqdm(enumerate(data)):
	    lst_cnt_diffs = [0] * (r * c)
	    for cnt in range(r * c):
	        lst_cnt_diffs[cnt] = compare_images(gen_imgs[cnt], k)
	    list_diffs[ind] = lst_cnt_diffs

	indx_more_sim_images = torch.argmin(torch.tensor(list_diffs).T, dim=1)
	sims_images = [data_train[ind] for ind in indx_more_sim_images]

	cnt = 0
	for i in range(r):
	    for j in range(c):
	        c_diff = float("inf")
	        c_img = None
	    
	        axs[i, j].imshow(sims_images[cnt][0], cmap="gray")
	        axs[i, j].axis("off")
	        cnt += 1

	plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model_path', type=str, default='output/weights_dcgan', metavar='N',
                        help='path of model')
    parser.add_argument('--lr_d', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--lr_g', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--model_name', type=str, default='AE', help='Model Name')
    parser.add_argument('--dataset_path', type=str, default='data', help='Dataset Path')
    parser.add_argument('--output_path', type=str, default='output', help='Dataset Path')
    parser.add_argument('--epochs', type=int, default=100, help='Num of epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--embedding_dim', type=int, default=100, help='batch size')
    parser.add_argument('--eval', action="store_true", help='train or eval')
    args = parser.parse_args()
    _init_()
    if not args.eval:
    	main(args)
    else:
    	test(args)
