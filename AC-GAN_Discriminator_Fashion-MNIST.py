class Discriminator(torch.nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = torch.nn.Sequential(
        
            
            torch.nn.Conv2d(
                in_channels= 1,
                out_channels=64 * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features=64 * 2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            
            torch.nn.Conv2d(
                in_channels=64 * 2,
                out_channels=64 * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features=64 * 4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

           
            torch.nn.Conv2d(
                in_channels=64 * 4,
                out_channels=64 * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features=64 * 8 ),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        # Categorical Classifier
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=64 * 8 * 3 * 3,
                out_features=nl, #number of classes
                bias=True
            ),
            torch.nn.Softmax()
        )
        
        # Real / Fake Classifier
        self.discr = torch.nn.Sequential(
            
            torch.nn.Conv2d(
                in_channels=ndf * 8,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False
            )
           
        )

    def forward(self, input):
        
        features = self.main(input)
        valid = self.discr(features).view(-1, 1)
        clf = self.clf(features.view(features.shape[0], -1))
        return valid, clf
