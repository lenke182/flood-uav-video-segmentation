# Entry-point for the segmentation task with frame interpolation and the s4GAN method

from base.cli import FlowLightningCLI
from flow.gan import FlowGANSemiSupervised, FlowGANDataModule

def cli_main():
    FlowLightningCLI(FlowGANSemiSupervised, FlowGANDataModule)


if __name__ == "__main__":
    cli_main()