class Discriminator(torch.nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels= 3,
                out_channels= 64 * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=64* 2
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),

            torch.nn.Conv2d(
                in_channels=64 * 2,
                out_channels=64 * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=64 * 4
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),

            torch.nn.Conv2d(
                in_channels= 64 * 4,
                out_channels= 64 * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=64 * 8
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            )
        )
            
        # Categorical Classifier
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=512*16*16,
                out_features = 2,
                bias=True
            ),
            torch.nn.Softmax(dim=1)
        )
        
            
        
        # Real / Fake Classifier
        self.police = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels= 64 * 8,
                out_channels=1,
                kernel_size=16,
                stride=1,
                padding=0,
                bias=False
            )
        )

    def forward(self, input):
        
        features = self.main(input)
        
        valid = self.police(features).view(-1, 1)
        clf = self.clf(features.view(features.shape[0], -1))
        return valid, clf
