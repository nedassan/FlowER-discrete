import os
import sys
import glob
import time
import datetime
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
import contextlib
import hashlib
from model.attn_encoder import AttnEncoderXL
from utils.data_utils import ReactionDataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from settings import Args
from model.flow_matching import DiscreteFlowMatcher 
from utils.train_utils import get_lr, grad_norm, log_rank_0, NoamLR, \
    param_count, param_norm, set_seed, setup_logger, log_args
from torch.nn.init import xavier_normal_
import torch.optim as optim

torch.set_printoptions(precision=4, profile="full", sci_mode=False, linewidth=10000)
np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True, linewidth=500)

def init_dist(args):
    if args.local_rank != -1:
        dist.init_process_group(backend=args.backend,
                                init_method='env://',
                                timeout=datetime.timedelta(minutes=10))
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = False

    if dist.is_initialized():
        logging.info(f"Device rank: {dist.get_rank()}")
        sys.stdout.flush()


def init_model(args):
    state = {}
    flow_model = DiscreteFlowMatcher(args) 
    
    if args.load_from:
        log_rank_0(f"Loading pretrained state from {args.load_from}")
        state = torch.load(args.load_from, map_location=torch.device("cpu"))
        pretrain_args = state["args"]
        pretrain_args.local_rank = args.local_rank

        graph_attn_model = AttnEncoderXL(pretrain_args)
        pretrain_state_dict = state["state_dict"]
        pretrain_state_dict = {k.replace("module.", ""): v for k, v in pretrain_state_dict.items()}
        graph_attn_model.load_state_dict(pretrain_state_dict)
        log_rank_0("Loaded pretrained model state_dict.")
    else:
        graph_attn_model = AttnEncoderXL(args)
        for p in graph_attn_model.parameters():
            if p.dim() > 1 and p.requires_grad:
                xavier_normal_(p, gain=1e-2)

    graph_attn_model.to(args.device)
    flow_model.to(args.device)
    
    if args.local_rank != -1:
        graph_attn_model = DDP(
            graph_attn_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
        log_rank_0("DDP setup finished")

    os.makedirs(args.model_path, exist_ok=True)

    return graph_attn_model, flow_model, state

def init_loader(args, dataset, batch_size: int, bucket_size: int = 1000,
                shuffle: bool = False, epoch: int = None, use_sort: bool =True):
    if use_sort: dataset.sort()
    if shuffle: dataset.shuffle_in_bucket(bucket_size=bucket_size)
    dataset.batch(
        batch_type=args.batch_type,
        batch_size=batch_size
    )

    if args.local_rank != -1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        if epoch is not None:
            sampler.set_epoch(epoch)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=lambda _batch: _batch[0],
        pin_memory=True
    )

    return loader

def get_optimizer_and_scheduler(args, model, state=None):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    scheduler = NoamLR(
        optimizer,
        model_size=args.emb_dim,
        warmup_steps=args.warmup_steps
    )

    if state and args.resume:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        log_rank_0("Loaded pretrained optimizer and scheduler state_dicts.")

    return optimizer, scheduler

def _optimize(args, model, optimizer, scheduler):
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
    optimizer.step()
    scheduler.step()
    g_norm = grad_norm(model)
    model.zero_grad(set_to_none=True)
    return g_norm

def main(args):
    if not args.load_from:
        checkpoints = glob.glob(os.path.join(args.model_path, "*.pt"))
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('.')[-2].split('_')[0]))[-1]
            args.load_from = latest_checkpoint
            args.resume = True
            log_rank_0(f"Auto-resuming from latest checkpoint: {latest_checkpoint}")

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = args.device

    init_dist(args)
    log_args(args, 'training')
    model, flow, state = init_model(args)
    total_step = state["total_step"] if state else 0
    log_rank_0(f"Number of parameters: {param_count(model)}")

    optimizer, scheduler = get_optimizer_and_scheduler(args, model, state)

    if dist.get_rank() == 0:
        config_dict = {k: v for k, v in vars(args).items() if not k.startswith('__')}

        if state and "run_id" in state:
            run_id = state["run_id"]
            log_rank_0(f"Resuming WandB run ID: {run_id}")
        else:
            run_id = wandb.util.generate_id()
            log_rank_0(f"Generated new WandB run ID: {run_id}")
        
        wandb.init(
            project=args.log_file, 
            name=args.exp_name, 
            id=run_id,
            resume="allow",
            config=config_dict,
            settings=wandb.Settings(init_timeout=300)
        )

    log_rank_0(f"Initializing training ...")
    log_rank_0(f"Loading data ...")
    if args.train_path.endswith('.pt'):
        train_dataset = ReactionDataset(args, smiles_list=None, cache_path=args.train_path)
        val_dataset = ReactionDataset(args, smiles_list=None, cache_path=args.val_path)
    else:
        with open(args.train_path, 'r') as train_o:
            train_smiles_list = train_o.readlines()
        with open(args.val_path, 'r') as val_o:
            val_smiles_list = val_o.readlines()
        train_dataset = ReactionDataset(args, train_smiles_list)
        val_dataset = ReactionDataset(args, val_smiles_list)

    accum = 0
    g_norm = 0
    losses = []
    active_losses, term_losses = [], []
    o_start = time.time()
    log_rank_0("Start training")

    for epoch in range(args.epoch):
        log_rank_0(f"Epoch: {epoch}")
        train_loader = init_loader(args, train_dataset,
                                batch_size=args.train_batch_size,
                                shuffle=True,
                                epoch=epoch)

        join_context = model.join() if (dist.is_initialized() and hasattr(model, "join")) else contextlib.nullcontext()
        
        with join_context:
            for train_batch in train_loader:
                if total_step > args.max_steps:
                    log_rank_0("Max steps reached, finish training")
                    if dist.get_rank() == 0: wandb.finish()
                    exit(0)

                train_batch.to(device)
                model.train()

                y = train_batch.src_token_ids
                y_len = train_batch.src_lens
                x0 = train_batch.src_matrices
                
                arrows = train_batch.src_arrows
                arrow_lens = train_batch.src_arrow_lens
                matrix_masks = train_batch.matrix_masks

                t = torch.rand(x0.shape[0]).type_as(x0)
                
                xt, target_rates, arrow_mask = flow.sample_conditional_pt(x0, arrows, arrow_lens, t)

                if hasattr(model, "module"):
                    model_inner = model.module  
                else:
                    model_inner = model

                y_emb = model_inner.id2emb(y)
                
                s_prop, t_prop, end_prop = model(y_emb, y_len, xt, t)

                loss, l_active, l_term = flow.compute_loss((s_prop, t_prop, end_prop), target_rates, arrows, arrow_lens, train_batch.matrix_masks)

                (loss / args.accumulation_count).backward()
                losses.append(loss.item())

                active_losses.append(l_active.item())
                term_losses.append(l_term.item())

                accum += 1
                if accum == args.accumulation_count:
                    g_norm = _optimize(args, model, optimizer, scheduler)
                    accum = 0
                    total_step += 1

                    if (total_step % args.log_iter == 0) and (dist.get_rank() == 0):
                        avg_loss = np.mean(losses)
                        avg_active = np.mean(active_losses)
                        avg_term = np.mean(term_losses)
                        curr_lr = get_lr(optimizer)
                        p_norm = param_norm(model)
                        
                        log_rank_0(f"Step {total_step}, loss: {avg_loss: .4f}, "
                                   f"p_norm: {p_norm: .4f}, g_norm: {g_norm: .4f}, "
                                   f"lr: {curr_lr: .6f}, "
                                   f"elapsed time: {time.time() - o_start: .0f}")
                        
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/active_loss": avg_active,
                            "train/term_loss": avg_term,
                            "train/grad_norm": g_norm,
                            "train/param_norm": p_norm,
                            "train/lr": curr_lr,
                            "train/step": total_step
                        })
                        
                        losses = []
                        active_losses = []
                        term_losses = []

                if (accum == 0) and (total_step > 0) and (total_step % args.eval_iter == 0):
                    from eval_multiGPU import get_predictions
                    
                    val_count = 50 
                    val_loader = init_loader(args, val_dataset,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            epoch=epoch)
                    
                    metrics = get_predictions(args, model, flow, val_loader, val_count)
                    
                    if dist.get_rank() == 0:
                        metrics = np.array(metrics)
                        
                        total_correct = np.sum(metrics[:, 0])
                        total_samples = np.sum(metrics)
                        
                        val_acc = total_correct / total_samples if total_samples > 0 else 0.0
                        
                        log_rank_0(f"Validation Acc: {(val_acc * 100): .2f}%")
                        
                        wandb.log({
                            "val/accuracy": val_acc,
                            "val/step": total_step
                        })
                    
                    model.train()

                if dist.get_rank() == 0:
                    if (accum == 0) and (total_step > 0) and (total_step % args.save_iter == 0):
                        n_iter = total_step // args.save_iter - 1
                        log_rank_0(f"Saving at step {total_step}")
                        state = {
                            "args": args,
                            "total_step": total_step,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict() if scheduler else None,
                            "run_id": wandb.run.id
                        }
                        torch.save(state, os.path.join(args.model_path, f"model.{total_step}_{n_iter}.pt"))

        if args.local_rank != -1:
            dist.barrier()
            
    log_rank_0("Epoch ended")
    if dist.get_rank() == 0: wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    args = Args
    logger = setup_logger(args, "train")
    args.local_rank = int(os.environ["LOCAL_RANK"]) if os.environ.get("LOCAL_RANK") else -1
    main(args)
