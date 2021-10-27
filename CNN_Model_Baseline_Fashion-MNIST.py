
class Baseline_CNN(torch.nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels= 1,
                out_channels=ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features= 64 * 2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Conv2d(
                in_channels=ndf * 2,
                out_channels=ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features= 64 * 4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Conv2d(
                in_channels=ndf * 4,
                out_channels=ndf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features=ndf * 8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        
        # Categorical Classifier
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(
                in_features= 64 * 8 * 3 * 3,
                out_features= 10,
                bias=True
            ),
            torch.nn.Softmax(dim=1)
        )
        
        

    def forward(self, input):
        features = self.main(input)
        clf = self.clf(features.view(features.shape[0], -1))
        return clf
