#!/usr/bin/env python

import os
import torch
from visuomotor_core.models.config import DiffusionConfig
from visuomotor_core.models.diffusion_policy import DiffusionPolicy
from visuomotor_core.dataset.zarr_dataset import ZarrDataset




def main(root, epoch, batch_size, out_dir):
    print(root)
    if root is None:
        print('[ERROR] Must specify path to data!')
        return

    repo_id = 'test'
    from pathlib import Path
    import matplotlib.pyplot as plt
    from datetime import datetime

    output_directory = Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    log_freq = 20
    checkpoint_freq = 2000

    cfg = DiffusionConfig()
    cfg.input_shapes["observation.state"] = [8]
    cfg.output_shapes["action"] = [8]
    # No usaremos use_gripper del config porque ya pusimos explícitamente 8 dimensiones
    
    base_dataset = ZarrDataset(root=root, n_obs_steps=cfg.n_obs_steps, h_act=cfg.horizon)
    
    dataloader = torch.utils.data.DataLoader(
        base_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    policy = DiffusionPolicy(cfg, dataset_stats=base_dataset.stats)
    policy.train()
    policy.to(device)

    # Optional torch.compile for extra step execution speed
    try:
        print("Compiling model for faster step execution...")
        policy = torch.compile(policy)
        print("Model compiled successfully!")
    except Exception as e:
        print(f"Skipping model compilation: {e}")

    # sqrt-scaled LR: appropriate for Adam when varying batch size
    base_lr = 1e-4
    lr = base_lr * (batch_size / 64) ** 0.5
    print(f"batch_size={batch_size}  lr={lr:.2e}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    n = len(base_dataset)
    steps_per_epoch = max(1, n // batch_size)
    epochs = epoch
    total_steps = epochs * steps_per_epoch
    print(f"Total epochs: {epochs} | Steps per epoch: {steps_per_epoch} | Total training steps: {total_steps}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    step = 0
    train_losses = []

    try:
        for epoch_idx in range(1, epochs + 1):
            for batch in dataloader:
                # Move batch to GPU
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output_dict = policy.forward(batch)
                    loss = output_dict["loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if step % log_freq == 0:
                    vram_mb = torch.cuda.max_memory_allocated() / 1e6
                    print(f"Epoch {epoch_idx}/{epochs} | Step {step}/{total_steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | VRAM(peak): {vram_mb:.0f}MB")
                    torch.cuda.reset_peak_memory_stats()

                if step > 0 and step % checkpoint_freq == 0:
                    orig_policy = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
                    orig_policy.save_pretrained(output_directory)
                    print(f"[checkpoint] saved at step {step}")

                step += 1
                train_losses.append(loss.item())
    except KeyboardInterrupt:
        print("\n[INFO] Entrenamiento interrumpido manualmente (KeyboardInterrupt). Guardando pesos actuales...")

    plt.plot(train_losses, label='Training loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Function')
    plt.legend()

    time_now = datetime.now()
    loss_path = 'outputs/losses/'
    if not os.path.exists(loss_path):
        os.mkdir(loss_path)

    name = loss_path + 'loss_' + time_now.strftime("%Y_%m_%d_%H_%M_%S") + '.png'
    plt.savefig(name, bbox_inches='tight', dpi=300)

    # Save the original uncompiled policy if compiled
    orig_policy = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
    orig_policy.save_pretrained(output_directory)
    print(f"\n[INFO] Modelo guardado exitosamente en: {output_directory}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train diffusion policy')
    parser.add_argument('--path', type=str, help='Data path')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (n_obs_steps=2 doubles the effective '
                         'image batch through the ResNet encoder; 224x224 crops OOM well before batch_size=512 on an '
                         '8GB GPU — start low and raise it while watching the printed VRAM figure)')
    parser.add_argument('--out_dir', type=str, default='outputs/train', help='Output directory')
    parsed_args = parser.parse_args()

    main(parsed_args.path, parsed_args.epoch, parsed_args.batch_size, parsed_args.out_dir)
