import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from kmm import kmm, get_kernel_width
from model import BERT_Regressor
from dataloader import ASAP_Dataloader
from transformers import AutoTokenizer  # Added this line


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--step', type=float, default=100, help='period of learning rate decay')
parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--num_val', type=int, default=30, help='total number of validation data')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training data')
parser.add_argument('--batch_size_val', type=int, default=30, help='batch size for validation data')
parser.add_argument('--num_epoch', type=int, default=400, help='total number of training epoch')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--device', type=str, default='mps', help='device to run the code')
parser.add_argument('--data_dir', type=str, default='./data/cross_prompt_attributes', help='directory to the ASAP data')
parser.add_argument('--target_prompt_id', type=int, default=1, help='target prompt id')
parser.add_argument('--model_name', type=str, default='distilbert/distilbert-base-uncased', help='pretrained model name')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.environ['PYTHONHASHSEED'] = '0'
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = args.device


def build_model():
    net = BERT_Regressor(args.model_name, num_labels=1).to(device)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step, gamma=args.gamma)

    return net, opt, scheduler


def main():
    # data loaders
    train_loader, val_loader, test_loader = ASAP_Dataloader(
        data_path=args.data_dir + f'/{args.target_prompt_id}/',
        tokenizer=AutoTokenizer.from_pretrained(args.model_name),
        attribute='score',
        batch_size=args.batch_size,
        val_batch_size=args.batch_size_val,
        max_length=512,
        num_val_samples=args.num_val,
    ).run()

    # define the model, optimizer, and lr decay scheduler
    net, opt, scheduler = build_model()

    # train the model
    test_acc = []

    for epoch in tqdm(range(args.num_epoch)):

        train_rmse_tmp = []
        test_rmse_tmp = []

        for data in train_loader:

            # weight estimation (we) step
            net.eval()

            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['score'].to(device)

            out_train = net(input_ids, attention_mask)
            l_tr = F.mse_loss(out_train.squeeze(), labels, reduction='none')

            d = next(iter(val_loader))

            val_labels = d['score'].to(device)
            val_input_ids = d['input_ids'].to(device)
            val_attention_mask = d['attention_mask'].to(device)

            out_val = net(val_input_ids, val_attention_mask)

            l_val = F.mse_loss(out_val.squeeze(), val_labels, reduction='none')
            l_tr_reshape = np.array(l_tr.detach().cpu()).reshape(-1, 1)
            l_val_reshape = np.array(l_val.detach().cpu()).reshape(-1, 1)

            # warm start
            if epoch < 1:
                coef = [1 for i in range(len(labels))]
            # obtain importance weights
            else:
                kernel_width = get_kernel_width(l_tr_reshape)
                coef = kmm(l_tr_reshape, l_val_reshape, kernel_width)

            w = torch.from_numpy(np.asarray(coef)).float().to(device)

            # weighted classification (wc) step
            net.train()
            out_train_wc = net(input_ids, attention_mask)
            l_tr_wc = F.mse_loss(out_train_wc.squeeze(), labels, reduction='none')
            l_tr_wc_weighted = torch.sum(l_tr_wc * w)

            opt.zero_grad()
            l_tr_wc_weighted.backward()
            opt.step()

            # train acc
            train_rmse_tmp.append(np.sqrt(mean_squared_error(labels.detach().cpu().numpy(), out_train_wc.detach().cpu().numpy())))

        train_accuracy_mean = np.mean(train_rmse_tmp)
        print("train rmse mean is", train_accuracy_mean)

        net.eval()
        # test acc
        for data in test_loader:
            test_input_ids = data['input_ids'].to(device)
            test_attention_mask = data['attention_mask'].to(device)
            test_label = data['score'].to(device)
            out_test = net(test_input_ids, test_attention_mask)
            test_rmse = np.sqrt(mean_squared_error(test_label.detach().cpu().numpy(), out_test.detach().cpu().numpy()))
            test_rmse_tmp.append(test_rmse)

        test_accuracy_mean = np.mean(test_rmse_tmp)
        print("test rmse mean is", test_accuracy_mean)
        test_acc.append(test_accuracy_mean)

        scheduler.step()


if __name__ == '__main__':
    main()
