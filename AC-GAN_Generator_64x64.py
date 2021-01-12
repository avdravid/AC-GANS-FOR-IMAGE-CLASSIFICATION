class Generator(torch.nn.Module):

    def __init__(self):

        super(Generator, self).__init__()
#1
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=nz + nl,      #nz=size of latent vector, nl = number of classes/labels
                out_channels=128 * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features=128 * 8), 
            torch.nn.ReLU(inplace=True),

#2   
            torch.nn.ConvTranspose2d(
                in_channels=128 * 8,
                out_channels=128 * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ), 
            torch.nn.BatchNorm2d(num_features=128 * 4),
            torch.nn.ReLU(inplace=True),

#3
            torch.nn.ConvTranspose2d(
                in_channels=128 * 4,
                out_channels=128 * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features=128 * 2),
            torch.nn.ReLU(inplace=True),
#4
            torch.nn.ConvTranspose2d(
                in_channels=128 * 2,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),

#5
            torch.nn.ConvTranspose2d(
                in_channels=128,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.Tanh()
            
        )

    def forward(self, inputs, condition):
        # concatenate latent vector with one-hot-encoded condition vector
        cat_inputs = torch.cat((inputs, condition), dim=1)
        
        cat_inputs = cat_inputs.unsqueeze(2).unsqueeze(3)

        return self.main(cat_inputs)
