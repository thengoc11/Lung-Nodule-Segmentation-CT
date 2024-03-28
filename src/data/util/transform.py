from torchvision import transforms


def make_transform():
    train_transforms = transforms.Compose([
        transforms.Resize(size=(352, 352)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30)])

    test_transforms = transforms.Compose([
        transforms.Resize(size=(352, 352)),
    ])
       



    return train_transforms, test_transforms