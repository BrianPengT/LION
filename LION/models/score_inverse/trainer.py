"""
This module implements the training loop for score-based models.

Author: Tianzhen Peng

References
----------
.. [Song2021] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., 
   Ermon, S., & Poole, B. (2021). "Score-Based Generative Modeling 
   through Stochastic Differential Equations." ICLR. https://openreview.net/forum?id=PxTIG12RRHS

.. [Song2022] Song, Y., Shen, L., Xing, L., & Ermon, S. (2022). 
   "Solving Inverse Problems in Medical Imaging with Score-Based 
   Generative Models." ICLR. https://openreview.net/forum?id=vaRCHVj0uGI
"""

from .loss import SMLoss

def train(model, sde, train_loader, optimizer, num_epochs, verbose_freq=0):
    """
    Train the score-based model.

    Args:
        model: the score-based model to be trained, which accepts (xt, t) as input and outputs the score function.
        sde: the SDE to be used for training.
        train_loader: a PyTorch DataLoader that provides batches of training data.
        optimizer: a PyTorch optimizer for updating the model parameters.
        num_epochs: the number of epochs to train.
        verbose_freq: the frequency (in epochs) to print the training loss. If set to 0, no printing will occur.

    Returns:
        A list of average losses for each epoch.
    """
    device = next(model.parameters()).device
    loss_fn = SMLoss(model, sde)

    losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x0 = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(x0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x0.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        if verbose_freq > 0 and (epoch + 1) % verbose_freq == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return losses