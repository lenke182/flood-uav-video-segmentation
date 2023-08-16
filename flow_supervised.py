# Entry-point for the segmentation task with frame interpolation and supervised learning

from base.cli import FlowLightningCLI
from flow.supervised import FlowSupervised, FlowSupervisedDataModule



def cli_main():
    FlowLightningCLI(FlowSupervised, FlowSupervisedDataModule)

if __name__ == "__main__":
    cli_main()