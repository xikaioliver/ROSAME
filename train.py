from ROSAME.models.rosame import *
from ROSAME.models.cv_gridworld import *

import torch
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

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
        if Ximg.dim()==6:
            Ximg = Ximg.unsqueeze(4).float()
        else:
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

    pred = pred_flat.reshape(-1, prop_num)
    target = target_flat.reshape(-1, prop_num)
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
    device,
    to_train=False,
):
    loss_final, acc_running, data_num = 0, 0, 0

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
        acc_running += compute_correctness(preds.data, label[:, :-1])
        data_num += preds.data.shape[0]

    if to_train:
        print(
            "Epoch {} TRAINING SET RESULTS: Average loss: {:.4f} Acc: {:.4f}".format(
                epoch, loss_final, acc_running / data_num
            )
        )
    else:
        print(
            "Epoch {} TESTING SET RESULTS: Average loss: {:.4f} Acc: {:.4f}".format(
                epoch, loss_final, acc_running / data_num
            )
        )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choice=["grid_block", "grid_gripper", "grid_logistics",
                                            "synth_block", "synth_hanoi", "synth_slide"])
    parser.add_argument("--gamma", type=float, default=10)
    parser.add_argument("--lambda_", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_schema", type=float, default=1e-3)
    parser.add_argument("--lr_gridcv_grid", type=float, default=1e-5)
    parser.add_argument("--lr_gridcv_mlp", type=float, default=1e-3)
    parser.add_argument("--lr_synth", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--trace_img_pth")
    parser.add_argument("--trace_label_pth")
    parser.add_argument("--trace_action_pth")
    parser.add_argument("--dataset_pth")
    parser.add_argument("--trace_num", type=int)
    parser.add_argument("--trace_len", type=int)
    parser.add_argument("--block_num", type=int, default=5)
    parser.add_argument("--ball_num", type=int, default=6)
    args = parser.parse_args()
    
    # Set up domain model and cv model.
    # Gather experiment data.
    if domain == "grid_block":
        block_num = args.block_num
        domain_model = get_domain_model_block(device)
        cv_model = CVGrid(GridConv(digit_class_num=block_num+1, input_channel=1),
                            block_dim=(block_num+1, block_num),
                            block_size=28, #MNIST images are 28x28
                            hidden_dim=128,
                            digit_class_num=block_num+1,
                            prop_dim=len(domain_model.propositions))
        data_transform = RearrangeColumn(block_num)
    elif domain == "grid_gripper":
        ball_num = args.ball_num
        domain_model = get_domain_model_gripper(device)
        cv_model = CVGrid(GridConv(digit_class_num=(ball_num+1)*2, input_channel=1),
                            block_dim=(4, ball_num),
                            block_size=28, #MNIST images are 28x28
                            hidden_dim=128,
                            digit_class_num=(ball_num+1)*2,
                            prop_dim=len(domain_model.propositions))
        data_transform = RearrangeBalls(ball_num)
    elif domain == "grid_logistics":
        domain_model = get_domain_model_logistics(device)
        digit_class_num = 35
        cv_model = CVGrid(GridConv(digit_class_num=digit_class_num, input_channel=3),
                            block_dim=(6, 6),
                            block_size=28, #MNIST images are 28x28
                            hidden_dim=256,
                            digit_class_num=digit_class_num,
                            prop_dim=len(domain_model.propositions))
        data_transform = RearrangeItems()
    elif domain == "synth_block":
        domain_model = get_domain_model_block(device)
        cv_model = torchvision.models.resnet18()
        cv_model.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(domain_model.propositions))
        )
        data_transform = transforms.Compose([transforms.Resize(64),
                                             transforms.RandomHorizontalFlip(0.5),])
    elif domain == "synth_hanoi":
        # WIP
        domain_model = get_domain_model_hanoi(device)
        cv_model = torchvision.models.resnet18()
        cv_model.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(domain_model.propositions))
        )
        data_transform = transforms.Resize(64)
    elif domain == "synth_slide":
        # WIP
        domain_model = get_domain_model_hanoi(device)
        cv_model = torchvision.models.resnet18()
        cv_model.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(domain_model.propositions))
        )
        data_transform = transforms.Resize(64)

    domain_model = domain_model.to(device)
    cv_model = cv_model.to(device)
    
    
    # Get Dataset
    if domain.startswith("grid"):
        trainset, testset = get_gridworld_datasets(args.trace_img_pth, args.trace_label_pth, args.trace_action_pth,
                                                   data_transform, 0.9, device)
    else:
        # WIP
        dataset = TraceImageDataset(args.dataset_pth, args.trace_len, transforms =data_transform)
        trainset, testset, _ = random_split(dataset, [args.trace_num, 100, len(dataset)-args.trace_num-100])
        
    train_loader = DataLoader(trainset, args.batch_size, shuffle=True)
    test_loader = DataLoader(testset, args.batch_size, shuffle=True)
    
    # Create optimizer
    parameters = []
    for schema in domain_model.action_schemas:
        parameters.append({'params': schema.parameters(), 'lr': lr_schema})
    if domain.startswith("grid"):
        parameters.extend([
            {'params': cv_model.mlp.parameters(), 'lr': lr_gridcv_mlp},
            {'params': cv_model.grid_convnet.parameters(), 'lr': lr_gridcv_grid},
            ])
    elif domain.startswith("synthesized"):
        pass
    optimizer = optim.Adam(parameters)