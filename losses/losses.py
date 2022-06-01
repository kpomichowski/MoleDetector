import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction="mean"):

        if reduction not in ("mean", "sum", "none"):
            raise RuntimeError(
                f"Inaproperiate reduction value. Available reductions: `mean`, `sum`, `none`."
            )

        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.tensor:
        logits = F.cross_entropy(input=input, target=target, reduction="none")
        pt = torch.exp(-logits)
        at = self.alpha.gather(0, target.data.view(-1))
        loss = at * (1 - pt) ** self.gamma * logits
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


def focal_loss(
    device: str,
    gamma: int = 2,
    reduction: str = "mean",
    alpha: torch.Tensor or None = None,
) -> FocalLoss:

    if alpha is not None:
        alpha = (
            alpha.to(device)
            if torch.is_tensor(alpha)
            else torch.tensor(alpha).to(device)
        )

    return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
