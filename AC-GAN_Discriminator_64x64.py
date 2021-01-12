class Discriminator(torch.nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = torch.nn.Sequential(
 #1
           torch.nn.Conv2d(
                in_channels=3,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

 #2
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128 * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features=128 * 2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

 #3
            torch.nn.Conv2d(
                in_channels=ndf * 2,
                out_channels=ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features=128 * 4),
            torch.nn.LeakyReLU(negative_slope=0.2,inplace=True),
                
                
  #4

            
            torch.nn.Conv2d(
                in_channels=128 * 4,
                out_channels=128* 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features=128* 8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)         
        
        )
        
        #Classifier
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=128 * 8 * 4 * 4,
                out_features=nl, #nl is number of classes/labels
                bias=True
            ),
            torch.nn.Softmax(dim=1)
        )
        
        # Real / Fake Classifier
        self.discr = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=128 * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        
        features = self.main(input)
        valid = self.discr(features).view(-1, 1)
        clf = self.clf(features.view(features.shape[0], -1))
        return valid, clf
