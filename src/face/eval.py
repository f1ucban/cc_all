import torch
from tqdm import tqdm
from face.configs.base import cfg
from src.utils.logger import setup_logger

logger = setup_logger("__validate__")


def val_epoch(model, val_dl, criterion, device, epoch, writer, time):
    model.eval()
    vloss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        pbar = tqdm(val_dl, desc=f"Epoch [{epoch+1}/{cfg.epochs}] - (Validate)")

        for i, (faces, lbls) in enumerate(pbar):
            faces, lbls = faces.to(device), lbls.to(device)
            mask = lbls != -1

            if torch.any(mask):
                faces, lbls = faces[mask], lbls[mask]
                logits, _ = model(faces, lbls)
                loss = criterion(logits, lbls)

                pred = logits.argmax(dim=1)
                correct += (pred == lbls).sum().item()
                total += lbls.size(0)
                vloss += loss.item() * lbls.size(0)

                step = epoch * len(val_dl) + i
                writer.add_scalar("1_batch/val_loss", loss.item(), step)
                writer.add_scalar("1_batch/val_acc", (pred == lbls).sum().item() / max(1, lbls.size(0)), step)

            pbar.set_postfix({"acc": f"{(correct / max(1, total)):.4f}"})

    avg_vloss = vloss / max(1, total)
    avg_vacc = correct / max(1, total)

    writer.add_scalar("2_epoch/val_loss", avg_vloss, epoch, walltime=time)
    writer.add_scalar("2_epoch/val_acc", avg_vacc, epoch, walltime=time)

    logger.info(
        f"Epoch [{epoch+1:03d}] - Loss: {(avg_vloss):.4f} | Acc: {(avg_vacc):.4f} |"
    )

    return avg_vacc
