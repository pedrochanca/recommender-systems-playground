import torch


def train_model(
    loader,
    model,
    loss_func,
    optimizer,
    scheduler,
    device: str,
    epochs: int = 1,
    log_every: int = 1000,
):
    """
    epochs: # nb. of times we go through the train set

    model.train():
        - Puts the model into "training mode"

        - It changes how some layers behave:
            1. Dropout layers (nn.Dropout)
                - In train() mode: randomly zero out some activations (adds noise,
                regularizes).
                - In eval() mode: no dropout, they pass everything through (but scaled
                appropriately during training).

            2. BatchNorm layers (nn.BatchNorm1d, nn.BatchNorm2d, etc.)
                (it fixes the "internal covariate shift" problem)
                - In train() mode: use the current batch's mean/variance and update
                running stats.
                - In eval() mode: use the stored running mean/variance (fixed
                statistics).
    """

    model.train()

    total_loss = 0
    total_samples = 0
    all_losses_list = []

    for epoch_i in range(epochs):
        for i, batch in enumerate(loader):
            users = batch["users"].to(device)
            items = batch["items"].to(device)
            targets = batch["targets"].to(device)

            # foward pass
            pred_target = model(users, items)
            true_target = targets.view(targets.size(0), -1).to(torch.float32)

            # reduction = "none" --> Vector [Batch_Size, 1]
            loss = loss_func(pred_target, true_target)

            # clears old gradients from previous iteration
            optimizer.zero_grad()

            # backpropagation: performs backward propragation
            # (fills param.grad for every parameter in model.parameters())

            # Manually calculate Mean for Backpropagation
            # The optimizer needs a single scalar to minimize.
            loss_scalar = loss.mean()
            loss_scalar.backward()
            # param update: uses the gradients in param.grads to update the parameters
            optimizer.step()

            # ---- Plot releated
            # Sum the vector directly for logging
            # loss.sum() adds up the squared errors of all X users in the batch
            total_loss += loss.sum().item()
            total_samples += users.size(0)
            if (i + 1) % log_every == 0:
                avg_loss = total_loss / total_samples
                print(
                    "Epoch: {} | Step: {} | Loss: {}".format(epoch_i, i + 1, avg_loss)
                )
                all_losses_list.append(avg_loss)

                # Reset
                total_loss = 0
                total_samples = 0

        if scheduler:
            scheduler.step()

    return model, all_losses_list


def evaluate_model(loader, model, loss_func, device: str):
    model.eval()  # Important: turns off dropout!

    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # Important: saves memory, no gradients
        for batch in loader:
            users = batch["users"].to(device)
            items = batch["items"].to(device)
            targets = batch["targets"].to(device).view(-1, 1).float()

            pred_target = model(users, items)
            true_target = targets.view(targets.size(0), -1).to(torch.float32)

            loss = loss_func(pred_target, true_target)

            total_loss += loss.sum().item()
            total_samples += users.size(0)

    return total_loss / total_samples
