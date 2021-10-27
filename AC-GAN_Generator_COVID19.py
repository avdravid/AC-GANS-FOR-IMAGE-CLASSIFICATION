class Generator(torch.nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        self.main = torch.nn.Sequential(
         
            torch.nn.ConvTranspose2d(
                in_channels=nz + nl,  #noise vector dimensionality + number of classes (2)
                out_channels= 64 * 16,
                kernel_size= 4,
                stride=1,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=64 * 16
            ),
            torch.nn.ReLU(
                inplace=True
            ),

            torch.nn.ConvTranspose2d(
                in_channels=64 * 16,
                out_channels=64 * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=64 * 8
            ),
            torch.nn.ReLU(
                inplace=True
            ),

            torch.nn.ConvTranspose2d(
                in_channels=64 * 8,
                out_channels=64 * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=64 * 4
            ),
            torch.nn.ReLU(
                inplace=True
            ),

           torch.nn.ConvTranspose2d(
                in_channels=64 * 4,
                out_channels=64*2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=64*2
            ),
            torch.nn.ReLU(
                inplace=True
            ),
            
             torch.nn.ConvTranspose2d(
                in_channels=64*2,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=64
            ),
            torch.nn.ReLU(
                inplace=True
            ),
            
            
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.Tanh()
 
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
