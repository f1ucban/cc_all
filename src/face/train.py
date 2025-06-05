import torch
from time import time
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau


from face.eval import val_epoch
from face.configs.base import cfg
from face.configs.utils import init_dm
from src.utils.logger import setup_logger
from face.configs.optim import param_groups
from dataloaders.face_loading import dataloaders


logger = setup_logger("__training__")


def train():
    torch.backends.cudnn.benchmark = True
    device, model = init_dm()
    torch.manual_seed(42)

    trn_dl, val_dl = dataloaders(batch_sz=cfg.bs).values()
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.lbl_smth)
    optimizer = optim.AdamW(param_groups(model), lr=cfg.lr / 2, weight_decay=cfg.wd)
    scaler = GradScaler()

    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=len(trn_dl) * 10,
    )
    plateau = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    accum_steps = 2
    best_acc, no_improv = 0.0, 0
    log = "runs/arcface_finetuning_v-"
    writer = SummaryWriter(log)

    for epoch in range(cfg.epochs):
        model.train()
        tloss, correct, total = 0.0, 0, 0

        pbar = tqdm(trn_dl, desc=f"Epoch [{epoch+1}/{cfg.epochs}] (Training)")
        for i, (faces, lbls) in enumerate(pbar):
            faces, lbls = faces.to(device), lbls.to(device)

            with autocast(device_type=device.type, enabled=True):
                logits, _ = model(faces, lbls)
                loss = criterion(logits, lbls) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(trn_dl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                warmup.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                mask = lbls != -1
                correct += (pred[mask] == lbls[mask]).sum().item()
                total += mask.sum().item()
                tloss += loss.item() * accum_steps * mask.sum().item()

                step = epoch * len(trn_dl) + i
                writer.add_scalar("1_batch/lr", optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("1_batch/train_loss", loss.item() * accum_steps, step)
                writer.add_scalar(
                    "1_batch/train_acc",
                    (pred[mask] == lbls[mask]).sum().item() / max(1, mask.sum().item()),
                    step,
                )

            pbar.set_postfix(
                {
                    "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                    "loss": f"{loss.item() * accum_steps:.4f}",
                    "acc": f"{(correct/total):.4f}",
                }
            )

        trn_loss = tloss / max(1, total)
        trn_acc = correct / max(1, total)

        logger.info(
            f"Epoch [{epoch+1:03d}] - Loss: {(trn_loss):.4f} | Acc: {(trn_acc):.4f} | LR: {(optimizer.param_groups[0]['lr']):.1e}"
        )

        writer.add_scalar("2_epoch/train_loss", trn_loss, epoch, walltime=time())
        writer.add_scalar("2_epoch/train_acc", trn_acc, epoch, walltime=time())

        val_acc = val_epoch(model, val_dl, criterion, device, epoch, writer, time())
        plateau.step(val_acc)

        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "config": cfg,
                },
                cfg.ckpt,
            )
            logger.info(
                f"Epoch [{epoch+1:03d}] - New BestModel| Acc: {best_acc:.4f} | Model Saved"
            )
            no_improv = 0
        else:
            no_improv += 1
            if no_improv >= cfg.patience or (
                no_improv >= cfg.patience // 2 and trn_loss < 0.05
            ):
                logger.info(
                    f"Epoch [{epoch+1:03d}] - NoImprovements: TrnStopped | EPOCH [{epoch+1:03d}]"
                )
                break

        torch.cuda.empty_cache()

    writer.close()

    logger.info(
        f"Training complete. Best Accuracy: {best_acc:.4f}. TensorBoard logs saved to {log}"
    )


if __name__ == "__main__":
    train()
