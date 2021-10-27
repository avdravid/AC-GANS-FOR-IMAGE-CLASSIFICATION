class Discriminator(torch.nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = torch.nn.Sequential(
        
            # state size. (3) x 128 x 128
            torch.nn.Conv2d(
                in_channels= 3,
                out_channels=ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=ndf * 2
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),

            # state size. (ndf*2) x 14 x 14
            torch.nn.Conv2d(
                in_channels=ndf * 2,
                out_channels=ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=ndf * 4
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),

            # state size. (ndf*4) x 7 x 7
            torch.nn.Conv2d(
                in_channels=ndf * 4,
                out_channels=ndf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=ndf * 8
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            )
        )
            
            
        #ndf*8 x 3 *3
        # Categorical Classifier
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=512*16*16,
                out_features=nl,
                bias=True
            ),
            torch.nn.Softmax(dim=1)
        )
        
            
        
        # Real / Fake Classifier
        self.police = torch.nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            torch.nn.Conv2d(
                in_channels=ndf * 8,
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
