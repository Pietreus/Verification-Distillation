import torch
from torch import nn
from torch.utils.data import Dataset


def prediction_confidence(prediction_tensor: torch.Tensor) -> torch.Tensor:
    return torch.softmax(prediction_tensor, dim=1).max(dim=1).values


def robustness_disparity(dataset: Dataset, teacher: nn.Module, student: nn.Module, grad_limit: float = 0.001) -> (
        torch.Tensor):

    data = dataset.features.clone().detach().requires_grad_(True)
    student.zero_grad()
    teacher.zero_grad()
    student_outputs = student(data)
    teacher_outputs = teacher(data)

    # ======== Confidences ==============

    teacher_conf = prediction_confidence(teacher_outputs)
    student_conf = prediction_confidence(student_outputs)

    # ========= Gradients ===============
    LCE = torch.nn.CrossEntropyLoss()
    CE_loss_T = LCE(student_outputs, dataset.labels)
    CE_loss_S = LCE(teacher_outputs, dataset.labels)
    teacher_grad = torch.norm(torch.autograd.grad(CE_loss_T, data, retain_graph=True, create_graph=True)[0], dim=1)
    student_grad = torch.norm(torch.autograd.grad(CE_loss_S, data, retain_graph=True, create_graph=True)[0], dim=1)

    # ========= DISPARITY CALCULATION =========
    # we assume the following: if the confidence is <= 0.5, the prediction WILL change for 2 classes,
    # and CAN for 3 or more, so the robustness radius for any prediction with conf. <= 0.5 is constant 0.
    # If this is not the case, the assumption of local linearity (for both student and teacher) implies
    # (for sufficiently large gradients), that the (minimum) distance to a decision boundary is obtained from the
    # equation conf.(x') = ||grad.|| * ||x-x'|| ± conf.(x) ==> rad(x) := ||x-x'|| = ||0.5 - conf.(x)||/||grad.||
    # With this formula, we can quantify in absolute terms how much less robust the teacher is than the student:
    # rad_S(x) - rad_T(x). When we consider the minimum value of this over all points, this gives a conservative
    # estimate of how much smaller the robustness of the teacher can be.
    # TODO: does this statement require additional probabilistic quantification?
    # TODO: not always conservative as is, should be somehow related to radius of student?

    # torch.div(torch.clip(student_conf - 0.5, min=0), student_grad + grad_limit)

    disparity = ((
            torch.div(torch.clip(teacher_conf - 0.5, min=0), teacher_grad + grad_limit)) /
            torch.div(torch.clip(student_conf - 0.5, min=0) + 10e-6, student_grad + grad_limit))

    return disparity, teacher_conf
