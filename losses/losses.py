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
        CE_loss = F.cross_entropy(input=input, target=target, reduction="none")
        target = target.type(torch.long)
        pt = torch.exp(-CE_loss)

        is_alpha = self.alpha is not None
        if is_alpha:
            assert (
                self.alpha.size() == pt.size() == CE_loss.size()
            ), "Batch size not equal to the num of weights in alpha."
        if self.reduction == "mean" and is_alpha:
            loss = (self.alpha * (1 - pt) ** self.gamma * CE_loss).mean()
        elif self.reduction == "sum" and is_alpha:
            loss = (self.alpha * (1 - pt) ** self.gamma * CE_loss).sum()
        elif self.reduction == "mean" and not is_alpha:
            loss = ((1 - pt) ** self.gamma * CE_loss).mean()
        elif self.reduction == "sum" and not is_alpha:
            loss = ((1 - pt) ** self.gamma * CE_loss).sum()

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
