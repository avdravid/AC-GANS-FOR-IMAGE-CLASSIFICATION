class Generator(torch.nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(
                in_channels=nz + nl,
                out_channels=ngf * 16,
                kernel_size= 4,
                stride=1,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=ngf * 16
            ),
            torch.nn.ReLU(
                inplace=True
            ),

            # state size. (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(
                in_channels=ngf * 16,
                out_channels=ngf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=ngf * 8
            ),
            torch.nn.ReLU(
                inplace=True
            ),

            # state size. (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d(
                in_channels=ngf * 8,
                out_channels=ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=ngf * 4
            ),
            torch.nn.ReLU(
                inplace=True
            ),

            # state size. (ngf*2) x 16 x 16
           

            
           torch.nn.ConvTranspose2d(
                in_channels=ngf * 4,
                out_channels=ngf*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=ngf*2
            ),
            torch.nn.ReLU(
                inplace=True
            ),
            
             torch.nn.ConvTranspose2d(
                in_channels=ngf*2,
                out_channels=ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=ngf
            ),
            torch.nn.ReLU(
                inplace=True
            ),
            
            
            torch.nn.ConvTranspose2d(
                in_channels=ngf,
                out_channels=nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, inputs, condition):
        # Concatenate Noise and Condition
        cat_inputs = torch.cat(
            (inputs, condition),
            dim=1
        )
        
        # Reshape the latent vector into a feature map.
        cat_inputs = cat_inputs.unsqueeze(2).unsqueeze(3)
       
        return self.main(cat_inputs)
