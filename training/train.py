import time
import math
import torch
import os
import numpy as np

class LLMTrainer:
    def __init__(self, model, optimizer, scheduler, train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.ptdtype = torch.bfloat16 if config.device != 'cpu' else torch.float32
        
        self.best_val_loss = float('inf')
        self.start_time = time.time()

    def _get_batch(self, loader, it):
        try:
            x, y = next(it)
        except (StopIteration, TypeError):
            it = iter(loader)
            x, y = next(it)
        return x.to(self.config.device), y.to(self.config.device), it

    @torch.no_grad()
    def evaluate(self, val_iter):
        self.model.eval()
        losses = []

        for _ in range(self.config.eval_steps):
            X, Y, val_iter = self._get_batch(self.val_loader, val_iter)
            with torch.autocast(device_type=str(self.config.device), dtype=self.ptdtype):
                _, loss = self.model(X, Y)
            losses.append(loss.item())

        self.model.train()
        return np.mean(losses), val_iter

    def train(self):
        print(f"Starting training on {self.config.device}...")
        last_loss = 0.0
        
        train_iter = iter(self.train_loader)
        val_iter = iter(self.val_loader)

        for step in range(1, self.config.max_steps + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            for _ in range(self.config.step_accum):
                X, Y, train_iter = self._get_batch(self.train_loader, train_iter)
                with torch.autocast(device_type=str(self.config.device), dtype=self.ptdtype):
                    _, loss = self.model(X, Y)
                    loss = loss / self.config.step_accum
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            last_loss += loss.item() * self.config.step_accum

            if step % self.config.log_interval == 0:
                avg_loss = last_loss / self.config.log_interval
                lr = self.scheduler.get_last_lr()[0]
                print(f"Step {step} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
                last_loss = 0.0

            if step % self.config.eval_interval == 0:
                val_loss, val_iter = self.evaluate(val_iter)
                print(f"--- Step {step} | Val Loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.2f} ---")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save("model_best.pth")

            if step % self.config.checkpoint_interval == 0:
                self._save(f"model_step_{step}.pth")

    def _save(self, filename):
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Saved: {path}")