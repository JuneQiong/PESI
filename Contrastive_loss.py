import torch
import torch.nn as nn

LARGE_NUM = 1e9

class Contrastive_loss(nn.Module):

    def calc_loss(self,
                                  hidden1: torch.Tensor,
                                  hidden2: torch.Tensor,
                                  hidden_norm: bool = True,
                                  temperature: float = 1.0):
        """
        hidden1: (batch_size, dim)
        hidden2: (batch_size, dim)
        """
        batch_size, hidden_dim = hidden1.shape

        if hidden_norm:
            hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
            hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.arange(0, batch_size).to(device=hidden1.device)
        masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(device=hidden1.device,
                                                                                                    dtype=torch.float)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = loss_a + loss_b
        return loss

