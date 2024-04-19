
import torch

def print_cuda_diagnostics():
    print("Torch was successfully imported")
    print("Version: " + str(torch.__version__))
    print("Cuda is available: " + str(torch.cuda.is_available()))

    print("Number of devices: " + str(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        print("---")
        print("Device " + str(i))
        print(torch.cuda.get_device_properties(i).name)
        try:
            print("Power (W): " + str(torch.cuda.power_draw(i)/1e3))
        except:
            print("Could not get power usage information!")


def get_available_gpus(max_power, max_gpus):
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        try:
            if torch.cuda.power_draw(i)/1e3 < max_power:
                available_gpus.append(i)
        except:
            available_gpus.append(i)
            
        if len(available_gpus) == max_gpus:
            break

    return available_gpus
        
if __name__ == "__main__":
    print_cuda_diagnostics()
    print("Available GPUs (20W threshold):")
    print(get_available_gpus(20,256))