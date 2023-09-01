import torch

MODEL_NAME = "06-23_17-21-40_DistMultLit_fb15k-237_model.pt"
MODEL_PATH = f"results/{MODEL_NAME}"

# Get number of parameters in model
def get_num_params(model):
    num_params = 0
    for weight_matrix in model.values():
        print(weight_matrix.shape)
        num_params += weight_matrix.numel()

    return num_params


if __name__ == '__main__':
    # Import model
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    print(model.keys())
    # Get number of parameters
    num_params = get_num_params(model)
    print(f"Number of parameters in model {MODEL_NAME}:\n{num_params}")