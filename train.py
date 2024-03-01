from ROSAME.models.rosame import *
from ROSAME.models.cv_gridworld import *

import torch
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split

import random


def get_domain_model_block(device):
    obj = Type("object", None)

    domain_model = Domain_Model(
        [
            Predicate("arm-empty", {}),
            Predicate("clear", {obj: 1}),
            Predicate("on-table", {obj: 1}),
            Predicate("holding", {obj: 1}),
            Predicate("on", {obj: 2}),
        ],
        [
            Action_Schema("pickup", {obj: 1}),
            Action_Schema("putdown", {obj: 1}),
            Action_Schema("stack", {obj: 2}),
            Action_Schema("unstack", {obj: 2}),
        ],
        device=device,
    )

    objects = {obj: ["block1", "block2", "block3", "block4", "block5"]}

    domain_model.ground(objects)
    return domain_model


def get_domain_model_gripper(device):
    base = Type("object", None)
    room = Type("room", base)
    ball = Type("ball", base)
    gripper = Type("gripper", base)

    domain_model = Domain_Model(
        [
            Predicate("at-robby", {room: 1}),
            Predicate("at", {ball: 1, room: 1}),
            Predicate("free", {gripper: 1}),
            Predicate("carry", {ball: 1, gripper: 1}),
        ],
        [
            Action_Schema("move", {room: 2}),
            Action_Schema("pick", {ball: 1, room: 1, gripper: 1}),
            Action_Schema("drop", {ball: 1, room: 1, gripper: 1}),
        ],
        device=device,
    )

    objects = {
        room: ["rooma", "roomb"],
        ball: ["ball1", "ball2", "ball3", "ball4", "ball5", "ball6"],
        gripper: ["left", "right"],
    }

    domain_model.ground(objects)
    return domain_model


def get_domain_model_logistics(device):
    base = Type("object", None)
    movable = Type("movable", base)
    location = Type("location", base)
    city = Type("city", base)
    obj = Type("obj", movable)
    transport = Type("transport", movable)
    truck = Type("truck", transport)
    airplane = Type("airplane", transport)
    airport = Type("airport", location)

    domain_model = Domain_Model(
        [
            Predicate("at", {movable: 1, location: 1}),
            Predicate("in", {obj: 1, transport: 1}),
            Predicate("in-city", {location: 1, city: 1}),
        ],
        [
            Action_Schema("load-truck", {obj: 1, truck: 1, location: 1}),
            Action_Schema("load-airplane", {obj: 1, airplane: 1, airport: 1}),
            Action_Schema("unload-truck", {obj: 1, truck: 1, location: 1}),
            Action_Schema("unload-airplane", {obj: 1, airplane: 1, airport: 1}),
            Action_Schema("drive-truck", {truck: 1, location: 2, city: 1}),
            Action_Schema("fly-airplane", {airplane: 1, airport: 2}),
        ],
        device=device,
    )

    objects = {
        location: ["city1-1", "city2-1"],
        city: ["city1", "city2"],
        obj: [
            "package1",
            "package2",
            "package3",
            "package4",
            "package5",
            "package6",
        ],
        truck: ["truckred", "trucklime"],
        airplane: ["planeblue", "planeyellow"],
        airport: ["city1-2", "city2-2"],
    }

    domain_model.ground(objects)
    return domain_model


class RearrangeColumn(object):
    def __init__(self, column_num):
        self.column_num = column_num

    def __call__(self, img):
        idx = torch.randperm(self.column_num)
        return torch.cat((img[:, [0]], img[:, 1:, idx]), 1)


class RearrangeBalls(object):
    def __init__(self, column_num):
        self.column_num = column_num

    def __call__(self, img):
        # img is steps * row * column * 28 * 28
        idx1 = torch.randperm(self.column_num)
        idx2 = torch.randperm(self.column_num)
        return torch.cat(
            (
                img[:, [0]],
                img[:, [1], idx1].unsqueeze(1),
                img[:, [2]],
                img[:, [3], idx2].unsqueeze(1),
            ),
            1,
        )


class RearrangeItems(object):
    def __call__(self, img):
        # img is steps * row * column * 3 * 28 * 28
        indices = [
            [
                [
                    (r, c)
                    for r in range(i * 3, i * 3 + 3)
                    for c in range(j * 3, j * 3 + 3)
                ]
                for j in range(2)
            ]
            for i in range(2)
        ]
        for i in range(2):
            for j in range(2):
                random.shuffle(indices[i][j])
        rows = []
        columns = []
        for r in range(6):
            for c in range(6):
                idx = indices[int(r / 3)][int(c / 3)].pop(0)
                rows.append(idx[0])
                columns.append(idx[1])
        return img[:, rows, columns, :, :, :].unflatten(1, (6, 6))


class CustomDataset(Dataset):
    def __init__(self, images, labels, actions, transform=None):
        self.images = images
        self.labels = labels
        self.actions = actions
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        action = self.actions[index]

        if self.transform is not None:
            img = self.transform(img)
        return img, label, action

    def __len__(self):
        return len(self.images)


def get_gridworld_datasets(
    img_pth, label_pth, action_pth, transform, train_frac, device
):
    with open(img_pth, "rb") as f:
        Ximg = torch.load(f)
        Ximg = Ximg.float()
    with open(label_pth, "rb") as f:
        Y = torch.load(f)
    with open(action_pth, "rb") as f:
        actions = torch.load(f)

    dataset = CustomDataset(Ximg, Y, actions, transform)
    trainset, testset = random_split(dataset, [train_frac, 1 - train_frac])
    return trainset, testset


@torch.no_grad()
def compute_correctness(pred_flat, target_flat):
    '''
    Expect input in the shape of (batch_size, trace_len, prop_num)
    '''
    trace_len = pred_flat.shape[1]
    prop_num = pred_flat.shape[2]

    pred = pred_flat.view(-1, prop_num)
    target = target_flat.view(-1, prop_num)
    pred = (pred>0.5).float()

    correct = torch.sum(torch.isclose(pred, target))

    return float(correct)/prop_num/trace_len


def run(
    epoch,
    cv_model,
    domain_model,
    optimizer,
    data_loader,
    gamma,
    lambda_,
    to_train=False,
):
    loss_final, acc_final = 0, 0

    if to_train:
        cv_model.train()  # Set model to training mode
    else:
        cv_model.eval()  # Set model to evaluate mode

    for i, (data, label, action) in enumerate(data_loader):
        data = data.to(device)
        label = label.to(device)
        action = action.to(device)
        trace_len = action.shape[1]
        flattened_data = data.flatten(start_dim=0, end_dim=1)

        with torch.set_grad_enabled(to_train):
            preds = cv_model(flattened_data)
            loss = 0
            # Domain model inference loss
            precon, addeff, deleff = domain_model.build(action.flatten())
            domain_preds = preds * (1 - deleff) + (1 - preds) * addeff
            # domain_preds = 1 - (1-preds*(1-deleff)) * (1-(1-preds)*addeff)
            validity_constraint = (1 - preds) * (precon)
            preds = preds.unflatten(0, (-1, trace_len))
            domain_preds = domain_preds.unflatten(0, (-1, trace_len))
            loss += F.mse_loss(domain_preds[:, :-1], preds[:, 1:], reduction="sum")
            loss += gamma * F.mse_loss(
                domain_preds[:, -1], label[:, -1], reduction="sum"
            )
            loss += F.mse_loss(
                validity_constraint,
                torch.zeros(
                    validity_constraint.shape,
                    dtype=validity_constraint.dtype,
                    device=device,
                ),
                reduction="sum",
            )
            # Add a prior
            loss += lambda_ * F.mse_loss(
                precon,
                torch.ones(precon.shape, dtype=precon.dtype, device=device),
                reduction="sum",
            )
            if to_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_final += loss.item()
        acc_final += compute_correctness(preds.data, label[:, :-1])

    if to_train:
        print(
            "Epoch {} TRAINING SET RESULTS: Average loss: {:.4f} Acc: {:.4f}".format(
                epoch, loss_final, acc_final / len(trainset)
            )
        )
    else:
        print(
            "Epoch {} TESTING SET RESULTS: Average loss: {:.4f} Acc: {:.4f}".format(
                epoch, loss_final, acc_final / len(testset)
            )
        )

    torch.cuda.empty_cache()
